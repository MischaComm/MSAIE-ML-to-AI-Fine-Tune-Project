# Deployment

**Live application:** https://REPLACE-WITH-RENDER-URL.onrender.com

**Health check:** https://REPLACE-WITH-RENDER-URL.onrender.com/health

**Expected startup delay:** the app runs on Render's free tier, which puts the service to sleep after about 15 minutes without traffic. The first request after that can take **30–60 seconds** while the instance wakes and loads the model. Later requests are fast. Each agent reply typically takes 5–20 seconds, depending on how many tools it calls.

No API key is needed to use the app; the AI model key is stored as a server-side secret.

**How to use it:** open the URL, then either pick a task on the left (sample files upload automatically) or upload your own CSV/JSON and type a request. Every reply shows an activity record listing the tools called, their arguments and their results.

**Hosting notes and limitations**
* Render free tier: 512 MB RAM, sleeps when idle, builds take several minutes.
* Uploads and results are kept in memory for 2 hours per browser session and are cleared when the service restarts or redeploys.
* Deploys happen only through GitHub Actions after the test suite passes. Render auto-deploy is turned off.
