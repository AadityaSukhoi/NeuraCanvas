# ⚠️ Run `app.py` instead of `oldermodel.py` for better performance, stability, and results.

`pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118`

## Optional (speed boost 🚀)

If you want faster inference and your GPU supports it:

`pip install xformers`


Then add:

`pipe.enable_xformers_memory_efficient_attention()`