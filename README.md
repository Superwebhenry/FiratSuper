# FiratSuper — אימון LoRA ל-Stable Diffusion

רפוזיטורי זה מכיל נוטבוק Google Colab מוכן לאימון LoRA על Stable Diffusion, עם שמירה אוטומטית ב-Google Drive.

## מה כלול

| קובץ | תיאור |
|------|--------|
| [`notebooks/SD_LoRA_Training_Colab.ipynb`](notebooks/SD_LoRA_Training_Colab.ipynb) | נוטבוק ראשי — הרץ את זה ב-Colab |
| [`configs/lora_sd15.toml`](configs/lora_sd15.toml) | תבנית הגדרות ל-SD 1.5 |
| [`configs/lora_sdxl.toml`](configs/lora_sdxl.toml) | תבנית הגדרות ל-SDXL |
| [`scripts/setup_drive_folders.py`](scripts/setup_drive_folders.py) | סקריפט ליצירת מבנה תיקיות ב-Drive |

## התחלה מהירה

1. פתח את [`notebooks/SD_LoRA_Training_Colab.ipynb`](notebooks/SD_LoRA_Training_Colab.ipynb) ב-[Google Colab](https://colab.research.google.com/)
2. **Runtime → Change runtime type → T4 GPU**
3. הרץ את התאים לפי הסדר
4. אשר חיבור ל-Google Drive כשתתבקש
5. העלה 10–30 תמונות אימון (ידנית ל-Drive או דרך התא בנוטבוק)
6. המתן לסיום האימון — ה-LoRA יישמר ב-Drive

## מבנה Google Drive

```
MyDrive/FiratSuper/
├── datasets/<project>/10_trigger/   ← תמונות + captions (.txt)
├── output/<project>/               ← checkpoints בזמן אימון
├── models/                         ← מודל בסיס (מורד פעם אחת)
├── loras/                          ← LoRA סופי (.safetensors)
└── logs/<project>/                 ← לוגים
```

## הגדרות חשובות (בתא 2 בנוטבוק)

| משתנה | ברירת מחדל | הסבר |
|--------|------------|------|
| `PROJECT_NAME` | `my_lora` | שם הפרויקט |
| `TRIGGER_WORD` | `sks person` | מילת טריגר בפרומпт |
| `MODEL_TYPE` | `sd15` | `sd15` או `sdxl` |
| `MAX_TRAIN_EPOCHS` | `10` | מספר epochs |
| `AUTO_CAPTION` | `True` | יצירת captions אוטומטית |

## שימוש ב-LoRA אחרי האימון

1. הורד את `MyDrive/FiratSuper/loras/<project>_lora.safetensors`
2. העתק ל-`models/Lora/` ב-Automatic1111, Forge, או ComfyUI
3. בפרומпт: `<TRIGGER_WORD>, ...`
4. משקל LoRA להתחלה: `0.6–0.9`

## דרישות

- חשבון Google (Drive + Colab)
- GPU ב-Colab (T4 מספיק ל-SD 1.5; SDXL דורש GPU חזק יותר)
- 10–30 תמונות אימון איכותיות

## הערות

- **אין לי גישה לחשבון Google שלך** — הנוטבוק רץ אצלך ב-Colab ואתה מאשר את Drive
- מודל הבסיס מורד פעם אחת ונשמר ב-Drive (~4GB ל-SD 1.5)
- האימון על T4 לוקח בדרך כלל 20–60 דקות (תלוי בכמות תמונות ו-epochs)

## טיפים לאימון טוב

- **גיוון** — זוויות, תאורה, רקעים שונים
- **איכות** — תמונות חדות, לא מטושטשות
- **Captions** — ודא שמילת הטריגר מופיעה בכל caption
- **Overfit** — אם התוצאה "נדבקת" מדי לתמונות, הורד epochs
