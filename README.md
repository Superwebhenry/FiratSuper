# FiratSuper — אימון LoRA ל-Flux (Colab A100)

הנתיב הנוכחי: **Flux.1 [dev] character LoRA** על Google Colab Pro **A100**.
SD 1.5 נכשל בגוף מלא (ידיים, עיניים, זהות). אל תאמן אותו שוב. אל תמחק את קבצי ה-LoRA הישנים.

## דאטאסט Flux v2 (Gate 1 GO)

| | |
|--|--|
| תיקייה 1 | [ADD_FLUX_PHOTOS](https://drive.google.com/drive/folders/1oLtTmwg2kt-Jn6zuci06ipRQoK6AOFVZ) — 31 זוגות |
| תיקייה 2 | [ADD_FLUX_CHEST](https://drive.google.com/drive/folders/1iEmUvagFQVJ2TArN_7ee4Af4TUti1hZw) — 7 שומרים (תמונה אחת בלי כיתוב, לא לאימון) |
| סה"כ | 38 זוגות תמונה + `.txt` |
| מילת טריגר | `ohwx woman` |
| קובץ פלט | `loras/lapetitemilf_flux_v2.safetensors` בלבד |
| מוגן | `lapetitemilf_flux` (v1) ו-`lapetitemilf_face` — אל תדרוס |

## מה כלול

| קובץ | תיאור |
|------|--------|
| [`notebooks/Flux_LoRA_Training_Colab.ipynb`](notebooks/Flux_LoRA_Training_Colab.ipynb) | נוטבוק Flux — הרץ את זה ב-Colab A100 |
| [`configs/train_lora_flux_a100.yaml`](configs/train_lora_flux_a100.yaml) | מתכון Ostris ai-toolkit |
| [`notebooks/SD_LoRA_Training_Colab.ipynb`](notebooks/SD_LoRA_Training_Colab.ipynb) | נוטבוק SD 1.5 ישן — לא לאמן מחדש |

## התחלה מהירה (Flux)

1. פתח את [Flux notebook ב-Colab](https://colab.research.google.com/github/Superwebhenry/FiratSuper/blob/cursor/sd-lora-colab-34fe/notebooks/Flux_LoRA_Training_Colab.ipynb)
2. **Runtime → Change runtime type → A100 GPU**. אל תבחר T4. אל תבחר TPU.
3. Chrome, חשבון Google אחד (`superweb.contact@gmail.com`). בפופאפ: **Allow ALL**
4. Hugging Face: אשר רישיון ל-[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) והדבק READ token
5. הרץ תאים 1 עד 10 לפי הסדר. תא 7 = בדיקה קצרה. תא 8 = אימון מלא.
6. ה-LoRA יישמר ב-`MyDrive/FiratSuper/loras/lapetitemilf_flux_v2.safetensors`
7. תא 10 מייצר תמונות בלי safety checker (זהות / הלבשה / עירום). מבוגרת בלבד.

## מבנה Google Drive

```
MyDrive/FiratSuper/
|-- ADD_FLUX_PHOTOS/                         ← 31 תמונות + captions
|-- ADD_FLUX_CHEST/                          ← 7 שומרים + captions (v2)
|-- datasets/lapetitemilf/10_ohwx_woman/     ← דאטאסט SD ישן, לא לגעת
|-- loras/lapetitemilf_flux_v2.safetensors    ← קובץ חדש (Flux v2)
|-- loras/lapetitemilf_flux.safetensors      ← v1, מוגן, אל תדרוס
|-- loras/lapetitemilf_face.safetensors      ← מוגן, אל תדרוס
|-- output/lapetitemilf/flux_eval_v2/        ← תמונות מתא 10
`-- keepers/
```

## הגדרות חשובות (Flux, תא 2)

| משתנה | ערך | הסבר |
|--------|-----|------|
| `PROJECT_NAME` | `lapetitemilf` | שם הפרויקט |
| `TRIGGER_WORD` | `ohwx woman` | מילת טריגר בפרומפט |
| `LORA_NAME` | `lapetitemilf_flux_v2` | שם הקובץ החדש בלבד |
| `TRAIN_STEPS` | `2000` | צעדי אימון |
| `NETWORK_DIM` | `16` | rank (מתכון Ostris) |

## שימוש אחרי האימון

1. באותו Colab: תא 10. פרומפט חייב להתחיל ב-`ohwx woman`
2. ב-ComfyUI: checkpoint של Flux.1 [dev] + הקובץ הזה. לא לטעון LoRA של SD 1.5 על Flux.
3. משקל LoRA: `0.8–1.0`

## דרישות

- Colab Pro עם **A100**
- Hugging Face token עם גישה ל-FLUX.1-dev
- תיקיות `ADD_FLUX_PHOTOS` + `ADD_FLUX_CHEST` ב-Drive (38 זוגות)

## SD 1.5 (ארכיון)

אל תריץ אימון ב-`SD_LoRA_Training_Colab.ipynb`. קובץ הפנים `lapetitemilf_face.safetensors` נשאר מוגן.
