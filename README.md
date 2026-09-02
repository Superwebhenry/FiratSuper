# FiratSuper — Flux LoRA (Colab A100)

האימון נגמר. **v2 נעול.** עכשיו רק מייצרים תמונות. אל תאמני SD. אל תדרסי LoRA.

## קובץ הייצור

`MyDrive/FiratSuper/loras/lapetitemilf_flux_v2.safetensors` (נעול)

גם נעולים: `lapetitemilf_flux` (v1), `lapetitemilf_face`.

## ייצור תמונות

1. פתחי את [המחברת ב-Colab](https://colab.research.google.com/github/Superwebhenry/FiratSuper/blob/cursor/sd-lora-colab-34fe/notebooks/Flux_LoRA_Training_Colab.ipynb)
2. **Runtime → Change runtime type → A100 GPU**
3. Chrome, חשבון אחד (`superweb.contact@gmail.com`). **Allow ALL**
4. Hugging Face READ token ל-[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
5. תאים **1, 2, 3**. ריצה חדשה: גם **4**. אחר כך **תא אחד מ־13 עד 40**. אל תריצי 5–9.
6. תאים 13–22 ו-28–37 = מקום אחד, מצלמה רחוקה, הורדה. תאים 23–27 ו-38–40 = סטים זוגיים / POV / facial.
7. עירום: LoRA 0.75, guidance 2.5. בלי scars. שומרים ל-`keepers/`
8. LoRA זכר **חדש** (לא דורס v2): תאים **41–45** פעם אחת, אחר כך תא **46 או 47 או 48** (5 תמונות close-up).

## דאטאסט Flux v2 (Gate 1 GO)

| | |
|--|--|
| תיקייה 1 | [ADD_FLUX_PHOTOS](https://drive.google.com/drive/folders/1oLtTmwg2kt-Jn6zuci06ipRQoK6AOFVZ) — 31 זוגות |
| תיקייה 2 | [ADD_FLUX_CHEST](https://drive.google.com/drive/folders/1iEmUvagFQVJ2TArN_7ee4Af4TUti1hZw) — 7 שומרים (תמונה אחת בלי כיתוב, לא לאימון) |
| סה"כ | 38 זוגות תמונה + `.txt` |
| מילת טריגר | `ohwx woman` |
| קובץ פלט | `loras/lapetitemilf_flux_v2.safetensors` — **נעול, אל תדרסי** |
| מוגן | `lapetitemilf_flux` (v1) ו-`lapetitemilf_face` |

## מה כלול

| קובץ | תיאור |
|------|--------|
| [`notebooks/Flux_LoRA_Training_Colab.ipynb`](notebooks/Flux_LoRA_Training_Colab.ipynb) | נוטבוק Flux — הרץ את זה ב-Colab A100 |
| [`configs/train_lora_flux_a100.yaml`](configs/train_lora_flux_a100.yaml) | מתכון Ostris ai-toolkit |
| [`notebooks/SD_LoRA_Training_Colab.ipynb`](notebooks/SD_LoRA_Training_Colab.ipynb) | נוטבוק SD 1.5 ישן — לא לאמן מחדש |

## התחלה מהירה (ייצור, לא אימון)

האימון כבר רץ. אל תריצי תאים 5–9.

1. פתחי את [Flux notebook ב-Colab](https://colab.research.google.com/github/Superwebhenry/FiratSuper/blob/cursor/sd-lora-colab-34fe/notebooks/Flux_LoRA_Training_Colab.ipynb)
2. **Runtime → Change runtime type → A100 GPU**. אל תבחר T4. אל תבחר TPU.
3. Chrome, חשבון Google אחד (`superweb.contact@gmail.com`). בפופאפ: **Allow ALL**
4. Hugging Face: אשר רישיון ל-[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) והדבק READ token
5. תאים 1, 2, 3. ריצה חדשה: גם 4. אחר כך תא אחד מ־13 עד 40. אופציונלי: 10–12.
6. הקובץ כבר ב-Drive: `MyDrive/FiratSuper/loras/lapetitemilf_flux_v2.safetensors`
7. כל תא 13–22 = 20 תמונות רחוקות, הורדה הדרגתית. מבוגרת בלבד.
8. LoRA זכר: אחרי 41–45, תא 46 או 47 או 48 (5 תמונות, שני LoRA).

## מבנה Google Drive

```
MyDrive/FiratSuper/
|-- ADD_FLUX_PHOTOS/                         ← 31 תמונות + captions
|-- ADD_FLUX_CHEST/                          ← 7 שומרים + captions (v2)
|-- ADD_HENRY_BODY_PHOTOS/                   ← תמונות גוף זכר (תא 41 בוחר 26)
|-- datasets/lapetitemilf/10_ohwx_woman/     ← דאטאסט SD ישן, לא לגעת
|-- loras/lapetitemilf_flux_v2.safetensors    ← נעול (ייצור)
|-- loras/henry_penis_flux_v1.safetensors     ← LoRA זכר (תאים 41-45)
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
| `LORA_NAME` | `lapetitemilf_flux_v2` | קובץ נעול לייצור |
| `TRAIN_STEPS` | `2000` | צעדי אימון |
| `NETWORK_DIM` | `16` | rank (מתכון Ostris) |

## שימוש אחרי האימון

1. באותו Colab: תא 10. פרומפט חייב להתחיל ב-`ohwx woman`
2. ב-ComfyUI: checkpoint של Flux.1 [dev] + הקובץ הזה. לא לטעון LoRA של SD 1.5 על Flux.
3. משקל LoRA: `0.8–1.0`

## RunComfy (זוגות)

אל תשתמשי בתבנית Flux.1 Inpaint — היא לא מחברת את ה-LoRA. גררי את הקובץ המוכן:

- [`workflows/couple_kontext_lora.json`](workflows/couple_kontext_lora.json) — קודם זה (בלי מסיכה)
- [`workflows/couple_fill_lora.json`](workflows/couple_fill_lora.json) — רק אם Kontext משנה את הגבר

הוראות קצרות: [`workflows/README.md`](workflows/README.md)

## LoRA זכר (תאים 41–48)

`lapetitemilf_flux_v2` נשאר נעול. אל תריצי 5–9.

1. אותה מחברת, A100, תאים 1–4.
2. תאים **41, 42, 43, 44, 45** — אימון `henry_penis_flux_v1` מ-`ADD_HENRY_BODY_PHOTOS` (26 תמונות אמיתיות, טריגר `hrmale`).
3. אחר כך **תא אחד**: 46 = vulva+penis, 47 = פה על הפין, 48 = facial עם semen ו-glans. 5 תמונות כל אחד.
4. שני ה-LoRA נטענים. הפרומפט קצר ומתחיל באיבר / semen כדי ש-CLIP לא יחתוך.

## דרישות

- Colab Pro עם **A100**
- Hugging Face token עם גישה ל-FLUX.1-dev
- תיקיות `ADD_FLUX_PHOTOS` + `ADD_FLUX_CHEST` ב-Drive (38 זוגות)
- `ADD_HENRY_BODY_PHOTOS` לאימון LoRA הזכר (תאים 41–45)

## SD 1.5 (ארכיון)

אל תריץ אימון ב-`SD_LoRA_Training_Colab.ipynb`. קובץ הפנים `lapetitemilf_face.safetensors` נשאר מוגן.
