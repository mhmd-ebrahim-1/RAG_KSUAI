# مساعد لائحة كلية الذكاء الاصطناعي (نسخة نهائية)

مساعد ذكي محلي للإجابة على أسئلة لائحة الكلية باستخدام RAG + Ollama.

## التشغيل السريع
1. تفعيل البيئة:
```powershell
& .\.venv\Scripts\Activate.ps1
```
2. تثبيت المتطلبات:
```powershell
d:/Downloads/files/.venv/Scripts/python.exe -m pip install -r requirements.txt
```
3. بناء الفهرس من الداتا:
```powershell
d:/Downloads/files/.venv/Scripts/python.exe build_clean_index.py
```
4. تشغيل Ollama:
```powershell
ollama serve
```
5. تشغيل الواجهة:
```powershell
d:/Downloads/files/.venv/Scripts/python.exe flask_app.py
```

الرابط:
- http://127.0.0.1:5000

## الملفات الأساسية
- `flask_app.py`: واجهة Flask والمنطق الخاص بالعرض.
- `rag_system.py`: محرك البحث والتوليد.
- `build_clean_index.py`: إعادة بناء الفهرس من `data.json`.
- `templates/index.html`: واجهة HTML.
- `static/style.css` و`static/app.js`: التنسيق والتفاعلات.
- `requirements.txt`: الاعتماديات.

## ميزات النسخة الحالية
- Retrieval محلي (FAISS + TF-IDF).
- توليد إجابة عربي عبر Ollama (`qwen2.5:1.5b-instruct`).
- تحسينات سرعة (cache لحالة Ollama + cache للأسئلة المتكررة).
- تنسيق إجابة ومصادر بشكل منظم.
- زر `مسح السابق` لمسح النتائج/الكاش.

## ملاحظات
- المشروع يعمل بالكامل محليًا.
- بعد تعديل `data.json` شغّل `build_clean_index.py` مرة أخرى.
