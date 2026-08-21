# FiratSuper — אימון LoRA ל-Stable Diffusion

רפוזיטורי זה מכיל נוטבוק Google Colab מוכן לאימון LoRA על Stable Diffusion, עם שמירה אוטומטית ב-Google Drive.

## דאטאסט מחובר

| | |
|--|--|
| תיקייה | [Lapetitemilf Model / dataset](https://drive.google.com/drive/folders/1FOwDPkzqjmOo0LPuNKmgJtK4YuWU9Pmi) |
| תמונות | 25 קבצי JPG |
| Captions | ייווצרו אוטומטית בנוטבוק |
| מילת טריגר | `ohwx woman` |
| סגנון יעד | בגדי ים והלבשה תחתונה (אופנה) |

## מה כלול

| קובץ | תיאור |
|------|--------|
| [`notebooks/SD_LoRA_Training_Colab.ipynb`](notebooks/SD_LoRA_Training_Colab.ipynb) | נוטבוק ראשי — הרץ את זה ב-Colab |
| [`configs/lora_sd15.toml`](configs/lora_sd15.toml) | תבנית הגדרות ל-SD 1.5 |
| [`configs/lora_sdxl.toml`](configs/lora_sdxl.toml) | תבנית הגדרות ל-SDXL |
| [`configs/dataset.json`](configs/dataset.json) | מזהה תיקיית Drive והגדרות דאטאסט |
| [`scripts/setup_drive_folders.py`](scripts/setup_drive_folders.py) | סקריפט ליצירת מבנה תיקיות ב-Drive |

## התחלה מהירה

1. פתח את [`notebooks/SD_LoRA_Training_Colab.ipynb`](notebooks/SD_LoRA_Training_Colab.ipynb) ב-[Google Colab](https://colab.research.google.com/)
2. **Runtime → Change runtime type → T4 GPU**
3. הרץ את התאים לפי הסדר
4. אשר חיבור ל-Google Drive כשתתבקש
5. התא של הדאטאסט יעתיק אוטומטית את 25 התמונות מתיקיית המקור
6. המתן לסיום האימון — ה-LoRA יישמר ב-Drive

## מבנה Google Drive

```
MyDrive/FiratSuper/
├── datasets/lapetitemilf/10_ohwx_woman/   ← עותק אימון + captions
├── output/lapetitemilf/                    ← checkpoints בזמן אימון
├── models/                                 ← מודל בסיס (מורד פעם אחת)
├── loras/                                  ← LoRA סופי (.safetensors)
└── logs/lapetitemilf/                      ← לוגים
```

## הגדרות חשובות (בתא 2 בנוטבוק)

| משתנה | ערך | הסבר |
|--------|-----|------|
| `PROJECT_NAME` | `lapetitemilf` | שם הפרויקט |
| `TRIGGER_WORD` | `ohwx woman` | מילת טריגר בפרומפט |
| `SOURCE_FOLDER_ID` | `1FOwDPkzqjmOo0LPuNKmgJtK4YuWU9Pmi` | תיקיית התמונות ב-Drive |
| `MODEL_TYPE` | `sd15` | `sd15` או `sdxl` |
| `MAX_TRAIN_EPOCHS` | `10` | מספר epochs |
| `STYLE_TAGS` | swimsuit / lingerie | תגיות אופנה ב-captions |

## שימוש ב-LoRA אחרי האימון

1. הורד את `MyDrive/FiratSuper/loras/lapetitemilf_lora.safetensors`
2. העתק ל-`models/Lora/` ב-Automatic1111, Forge, או ComfyUI
3. בפרומפט: `ohwx woman, swimsuit, ...` או `ohwx woman, lingerie, ...`
4. משקל LoRA להתחלה: `0.6–0.9`

## דרישות

- חשבון Google (Drive + Colab)
- GPU ב-Colab (T4 מספיק ל-SD 1.5; SDXL דורש GPU חזק יותר)
- תיקיית התמונות כבר מחוברת ב-Drive

## טיפים לאימון טוב

- **גיוון** — זוויות, תאורה, רקעים שונים
- **איכות** — תמונות חדות, לא מטושטשות
- **Captions** — ודא שמילת הטריגר מופיעה בכל caption
- **Overfit** — אם התוצאה "נדבקת" מדי לתמונות, הורד epochs
