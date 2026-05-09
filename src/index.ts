import type {
  Plugin,
  ContentConverter,
  ConversionInput,
  ConversionResult,
  ConverterStageDescriptor,
} from '@yandu/types';
import { spawn } from 'child_process';
import { writeFile, mkdir } from 'fs/promises';
import path from 'path';

interface PaddleOCRResult {
  text: string;
  confidence: number;
  pages: Array<{ page: number; text: string; confidence: number }>;
}

interface LayoutRegion {
  type: 'text' | 'title' | 'figure' | 'table' | 'footer' | 'header' | 'reference';
  bbox: [number, number, number, number];
  text?: string;
  confidence?: number;
  html?: string;
}

interface LayoutPage {
  page: number;
  regions: LayoutRegion[];
}

interface PPStructureResult {
  pages: LayoutPage[];
  error?: { message: string; trace?: string };
}

async function findPython(): Promise<string | null> {
  const candidates = ['python3', 'python'];
  for (const cmd of candidates) {
    try {
      const code = await new Promise<number>((resolve) => {
        const p = spawn(cmd, ['--version'], { stdio: 'ignore' });
        p.on('exit', (c) => resolve(c ?? 1));
        p.on('error', () => resolve(1));
      });
      if (code === 0) return cmd;
    } catch {
      // continue
    }
  }
  return null;
}

async function runPythonScript(
  pythonPath: string,
  script: string,
  args: string[],
  opts: {
    timeout?: number;
    onProgressLine?: (data: unknown) => void;
  } = {}
): Promise<string> {
  const { timeout, onProgressLine } = opts;

  return new Promise((resolve, reject) => {
    let stdout = '';
    let stderr = '';
    let resolved = false;

    const child = spawn(pythonPath, ['-c', script, ...args], {
      env: process.env,
    });

    child.stdout?.on('data', (chunk: Buffer) => {
      const text = chunk.toString();
      if (onProgressLine) {
        for (const line of text.split('\n')) {
          const trimmed = line.trim();
          if (trimmed.startsWith('PROGRESS:')) {
            try {
              const data = JSON.parse(trimmed.slice('PROGRESS:'.length));
              onProgressLine(data);
            } catch {
              // ignore malformed PROGRESS lines
            }
          } else if (line) {
            stdout += line;
          }
        }
      } else {
        stdout += text;
      }
    });

    child.stderr?.on('data', (chunk: Buffer) => {
      stderr += chunk.toString();
    });

    child.on('exit', (code) => {
      if (resolved) return;
      resolved = true;
      if (code !== 0) {
        reject(new Error(`Python script failed (exit ${code}): ${stderr.slice(0, 2000)}`));
        return;
      }
      resolve(stdout.trim());
    });

    child.on('error', (err) => {
      if (resolved) return;
      resolved = true;
      reject(new Error(`Failed to spawn Python: ${err.message}`));
    });

    if (timeout && timeout > 0) {
      setTimeout(() => {
        if (resolved) return;
        resolved = true;
        child.kill('SIGTERM');
        setTimeout(() => child.kill('SIGKILL'), 5000);
        reject(new Error(`Python script timed out after ${timeout}ms`));
      }, timeout);
    }
  });
}

class PaddleOCRConverter implements ContentConverter {
  id = 'converter.pdf.paddleocr';
  inputFormats = ['application/pdf'];

  settingsSchema = {
    type: 'object',
    properties: {
      keepOriginal: { type: 'boolean', default: true },
      enableLayoutAnalysis: {
        type: 'boolean',
        default: true,
        description: 'Enable PP-Structure layout analysis for table/figure/title detection',
      },
      useGpu: { type: 'boolean', default: false },
      lang: { type: 'string', default: 'ch' },
      timeoutMs: { type: 'number', default: 300_000 },
    },
  };

  getStages(): ConverterStageDescriptor[] {
    return [
      { id: 'ocr', label: 'Text Recognition', weight: 0.60 },
      { id: 'layout', label: 'Layout Analysis', weight: 0.25 },
      { id: 'assemble', label: 'Assemble Result', weight: 0.15 },
    ];
  }

  async convert(input: ConversionInput): Promise<ConversionResult> {
    const { source, outputDir, onProgress } = input;
    const settings = (input.settings || {}) as {
      enableLayoutAnalysis?: boolean;
      useGpu?: boolean;
      lang?: string;
      timeoutMs?: number;
    };

    const pythonPath = await findPython();
    if (!pythonPath) {
      throw new Error('Python not found. Please install Python 3 and ensure it is in PATH.');
    }

    await mkdir(outputDir, { recursive: true });

    let pdfPath: string;
    if (typeof source === 'string') {
      pdfPath = source;
    } else {
      const tempPath = path.join(outputDir, 'temp_input.pdf');
      const buffer = Buffer.from(await source.arrayBuffer());
      await writeFile(tempPath, buffer);
      pdfPath = tempPath;
    }

    const useGpu = settings.useGpu ?? false;
    const lang = settings.lang ?? 'ch';
    const timeout = settings.timeoutMs ?? 300_000;

    // OCR phase
    onProgress?.({ stageId: 'ocr', stageProgress: 0 });
    const gpuFlag = useGpu ? 'True' : 'False';
    const ocrScript = `
import sys, json
try:
    from paddleocr import PaddleOCR
    from pdf2image import convert_from_path
    import numpy as np

    ocr = PaddleOCR(use_angle_cls=True, lang="${lang}", use_gpu=${gpuFlag}, show_log=False)
    pages_result = []
    images = convert_from_path(sys.argv[1])
    total_pages = len(images)
    for i, img in enumerate(images):
        img_array = np.array(img)
        result = ocr.ocr(img_array, cls=True)
        page_text = ""
        total_conf = 0
        count = 0
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    conf = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 1.0
                    page_text += text + "\\n"
                    total_conf += conf
                    count += 1
        avg_conf = total_conf / count if count > 0 else 0
        pages_result.append({"page": i + 1, "text": page_text.strip(), "confidence": avg_conf})
        print("PROGRESS:" + json.dumps({"page": i + 1, "total": total_pages}))
        sys.stdout.flush()

    total_text = "\\n\\n".join(p["text"] for p in pages_result)
    avg_conf = sum(p["confidence"] for p in pages_result) / len(pages_result) if pages_result else 0
    print(json.dumps({"text": total_text, "confidence": avg_conf, "pages": pages_result}))
except Exception as e:
    print(json.dumps({"error": str(e)}), file=sys.stderr)
    sys.exit(1)
`;

    const ocrStdout = await runPythonScript(pythonPath, ocrScript, [pdfPath], {
      timeout,
      onProgressLine: (data) => {
        const d = data as { page?: number; total?: number };
        if (typeof d.page === 'number' && typeof d.total === 'number') {
          onProgress?.({
            stageId: 'ocr',
            stageProgress: (d.page / d.total) * 100,
            detail: `Page ${d.page}/${d.total}`,
            overallProgress: Math.round((d.page / d.total) * 60),
          });
        }
      },
    });

    const ocrResult = JSON.parse(ocrStdout) as PaddleOCRResult;
    if ('error' in (ocrResult as unknown as Record<string, unknown>)) {
      throw new Error(`PaddleOCR error: ${(ocrResult as unknown as { error: string }).error}`);
    }

    onProgress?.({ stageId: 'ocr', stageProgress: 100, overallProgress: 60 });

    // Layout analysis phase
    const enableLayout = settings.enableLayoutAnalysis ?? true;
    onProgress?.({ stageId: 'layout', stageProgress: 0, overallProgress: 60 });

    let layoutResult: PPStructureResult = { pages: [] };
    if (enableLayout) {
      const layoutScript = `
import sys, json
try:
    from paddleocr import PaddleOCR
    from pdf2image import convert_from_path
    import numpy as np

    table_engine = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=${gpuFlag}, show_log=False, structure=True)
    pages_result = []
    images = convert_from_path(sys.argv[1])
    for i, img in enumerate(images):
        img_array = np.array(img)
        result = table_engine(img_array)
        regions = []
        if result:
            for line in result:
                if not line:
                    continue
                reg_type = line.get('type', 'text').lower()
                bbox = line.get('bbox', [0, 0, 0, 0])
                res = line.get('res', '')
                region = {"type": reg_type, "bbox": bbox}
                if reg_type == 'table':
                    if isinstance(res, dict) and 'html' in res:
                        region["html"] = res['html']
                    elif isinstance(res, str):
                        region["text"] = res
                elif isinstance(res, (list, tuple)) and len(res) >= 2:
                    texts = []
                    total_conf = 0
                    count = 0
                    for item in res[1]:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            texts.append(str(item[0]))
                            total_conf += float(item[1]) if isinstance(item[1], (int, float)) else 1.0
                            count += 1
                    region["text"] = "\\n".join(texts)
                    region["confidence"] = total_conf / count if count > 0 else 0
                else:
                    region["text"] = str(res) if res else ''
                regions.append(region)
        pages_result.append({"page": i + 1, "regions": regions})
    print(json.dumps({"pages": pages_result}))
except Exception as e:
    import traceback
    print(json.dumps({"error": str(e), "trace": traceback.format_exc()}), file=sys.stderr)
    sys.exit(1)
`;
      try {
        const layoutStdout = await runPythonScript(pythonPath, layoutScript, [pdfPath], { timeout });
        const parsed = JSON.parse(layoutStdout) as PPStructureResult & { error?: string; trace?: string };
        if (parsed.error) {
          console.warn(`[PaddleOCRConverter] Layout analysis degraded: ${parsed.error}`);
        } else {
          layoutResult = parsed;
        }
      } catch (e) {
        console.warn(`[PaddleOCRConverter] Layout analysis failed: ${e instanceof Error ? e.message : String(e)}`);
      }
    }

    onProgress?.({ stageId: 'layout', stageProgress: 100, overallProgress: 85 });

    // Assemble Markdown
    const markdownLines: string[] = [];
    const figures: Array<{ id: string; caption: string; page: number }> = [];

    for (const page of ocrResult.pages) {
      markdownLines.push(`# Page ${page.page}\n`);
      const layoutPage = layoutResult.pages.find((p) => p.page === page.page);
      if (layoutPage && layoutPage.regions.length > 0) {
        for (const region of layoutPage.regions) {
          if (region.type === 'table' && region.html) {
            markdownLines.push('\n<!-- table -->\n');
            markdownLines.push(region.html);
            markdownLines.push('\n<!-- /table -->\n');
          } else if (region.type === 'figure') {
            const figId = `fig_${page.page}_${figures.length + 1}`;
            figures.push({ id: figId, caption: region.text || '', page: page.page });
            markdownLines.push(`\n![${region.text || 'Figure'}](${figId})\n`);
          } else if (region.type === 'title') {
            markdownLines.push(`\n## ${region.text || ''}\n`);
          } else if (region.text) {
            markdownLines.push(region.text);
            markdownLines.push('\n\n');
          }
        }
      } else {
        markdownLines.push(page.text);
      }
      markdownLines.push('\n---\n');
    }
    const markdown = markdownLines.join('\n');

    const markdownPath = path.join(outputDir, 'output.md');
    await writeFile(markdownPath, markdown);

    // Generate mapping.yaml (simple YAML without external dep)
    let mdOffset = 0;
    const mappings = ocrResult.pages.map((page) => {
      const header = `# Page ${page.page}\n\n`;
      const content = page.text + '\n\n---\n\n';
      const totalLen = header.length + content.length;
      const entry = {
        mdOffset,
        mdLength: totalLen,
        sourcePosition: {
          format: 'application/pdf' as const,
          page: page.page,
          bbox: [0, 0, 0, 0] as [number, number, number, number],
        },
      };
      mdOffset += totalLen;
      return entry;
    });

    const yamlLines = [
      `version: 1`,
      `sourceFormat: application/pdf`,
      `sourcePath: ${JSON.stringify(pdfPath)}`,
      'mappings:',
      ...mappings.map((m) =>
        `  - mdOffset: ${m.mdOffset}\n    mdLength: ${m.mdLength}\n    sourcePosition:\n      format: ${m.sourcePosition.format}\n      page: ${m.sourcePosition.page}\n      bbox: [${m.sourcePosition.bbox.join(', ')}]`
      ),
      'figures:',
      ...figures.map((f) =>
        `  - id: ${f.id}\n    path: ""\n    sourcePosition:\n      format: application/pdf\n      page: ${f.page}\n      bbox: [0, 0, 0, 0]\n    caption: ${JSON.stringify(f.caption)}`
      ),
      'equations: []',
      'citations: []',
    ];

    const mappingPath = path.join(outputDir, 'paper.mapping.yaml');
    await writeFile(mappingPath, yamlLines.join('\n'));

    if (typeof source !== 'string' && pdfPath.includes('temp_input')) {
      const { unlink } = await import('fs/promises');
      await unlink(pdfPath).catch(() => {});
    }

    onProgress?.({ stageId: 'assemble', stageProgress: 100, overallProgress: 100 });

    return {
      markdown,
      markdownPath,
      mappingPath,
      assets: [],
    };
  }
}

export default {
  name: '@yandu/plugin-converter-paddleocr',
  version: '1.0.0',
  register(system) {
    const converter = new PaddleOCRConverter();
    system.capabilities.register(
      { type: 'converter', id: converter.id, name: 'PaddleOCR PDF Converter' },
      converter,
    );
  },
} satisfies Plugin;
