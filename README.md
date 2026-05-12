# yandu-plugin-converter-paddleocr

Yandu PDF converter plugin using PaddleOCR.

## Requirements

- Python 3
- paddleocr (`pip install paddleocr pdf2image`)
- poppler (for pdf2image)

## Registration

```typescript
import plugin from 'yandu-plugin-converter-paddleocr';
plugin.register(system);
```
