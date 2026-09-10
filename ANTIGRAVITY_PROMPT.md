# Prompt for Google Antigravity

Copy everything in the code block below and paste it into Antigravity as your
task. Do this **after** unzipping `hdd-failure-predictor.zip` somewhere on
your machine and opening that folder in Antigravity as the workspace — the
prompt assumes the project already exists and just needs finishing, not
building from a blank folder.

---

```
I have an existing Python/Streamlit project in this workspace called
"Predicting Drive Failure" — a hard-drive failure prediction tool for an IT
capstone project. It's already fully built and tested: a data pipeline, a
trained scikit-learn classifier, a Cox Proportional Hazards survival model
(lifelines), SHAP explainability, and THREE Streamlit dashboards (Fleet
Overview, Operator Lookup, Optimization Lab) styled with a custom dark theme
and two purposeful 3D visualizations (a Three.js drive rack and a Plotly 3D
cost surface). Read README.md first for full context before doing anything —
it explains WHY this project is built the way it is (early-warning labeling
instead of same-day diagnosis, survival analysis instead of just yes/no,
per-prediction SHAP, a cost-optimization surface), which matters because my
professor specifically warned that this project needs to be genuinely
different from prior work in this area, not just cosmetically different.

The ONE thing it's currently missing: it ships with a simulated sample
dataset (data/sample_drive_stats.csv) instead of real data, because it was
built in a sandboxed environment that couldn't reach external file hosts.
Your job is to finish that, verify everything still works, and ship it.
Do NOT simplify the project back down to a plain classifier+dashboard while
"cleaning up" — the survival model, SHAP, and cost simulator are the point.
Do these steps in order:

1. SET UP THE ENVIRONMENT
   - Create a Python virtual environment and install requirements.txt.
   - Confirm Python 3.10+ is being used.
   - If `pip install lifelines` fails with "AttributeError: install_layout"
     (a known Debian/Ubuntu setuptools quirk, unrelated to this project),
     retry with: SETUPTOOLS_USE_DISTUTILS=stdlib pip install lifelines

2. GET REAL DATA
   - Go to https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data
     and download ONE recent quarterly ZIP of the Drive Stats dataset.
   - Unzip it into data/raw/ (you should end up with ~90 daily CSV files).
   - Look at the "model" column in a couple of those CSVs and pick ONE
     high-volume drive model to focus on (mixing models with different SMART
     baselines makes the signal noisier — pick whichever model has the most
     rows).
   - Run all four pipeline scripts, in this order:
       python scripts/prepare_data.py --raw-dir data/raw --models <the model you picked>
       python scripts/train_model.py
       python scripts/prepare_survival_data.py --raw-dir data/raw --models <the same model>
       python scripts/train_survival_model.py
   - Confirm scripts/prepare_data.py wrote data/provenance.json with
     "source": "real_backblaze_download" — this is what makes the
     dashboard's data-authenticity banner switch from the "simulated" warning
     to a "real data" confirmation automatically. Don't hand-edit that banner
     text anywhere — it's supposed to read provenance.json, not be hardcoded.
   - Report the real precision/recall/F1 from train_model.py AND the real
     concordance index from train_survival_model.py. These will likely be
     lower and noisier than the demo numbers on synthetic data — that's
     expected and correct, don't try to make them look better by changing
     the train/test split until the numbers flatter the model.

3. VERIFY NOTHING BROKE
   - Run the app locally: streamlit run app.py
   - Open http://localhost:8501 and click through all three dashboards: Home
     banner should say "Running on real Backblaze drive-day data"; Fleet
     Overview should load with the 3D rack visible and "Step forward" /
     "Live mode" working; Operator Lookup should show a SHAP bar chart (not
     an error) for all three input sources; Optimization Lab should show a
     survival curve and a 3D cost surface that responds to the sliders.
   - Also run this scripted check and paste me the output — it actually
     executes each page's Python (not just checks the URL responds):

     python3 -c "
     from streamlit.testing.v1 import AppTest
     for path in ['app.py', 'pages/1_Fleet_Overview.py', 'pages/2_Operator_Lookup.py', 'pages/3_Optimization_Lab.py']:
         at = AppTest.from_file(path, default_timeout=30)
         at.run()
         print(path, '->', at.exception[0] if at.exception else 'OK')
     "

   - Fix any exceptions before moving on. Do not proceed to deployment with
     a broken page.

4. PUSH TO GITHUB
   - Initialize git if this folder isn't already a repo.
   - Make sure .gitignore is respected (it already excludes data/raw/*.csv —
     don't commit the raw multi-GB download, only the small processed files).
   - Create a new GitHub repository named "hdd-failure-predictor" (ask me to
     create the empty repo on github.com first if you don't have gh CLI
     access configured, then use the URL I give you).
   - Commit everything with a clear message and push to the main branch.

5. DEPLOY ON RENDER
   - This repo already has a render.yaml Blueprint file, so the easiest path
     is: I'll go to render.com myself, choose "New -> Blueprint", and point
     it at the GitHub repo we just pushed — walk me through that if I ask,
     but you don't need browser/account access to do this part since it
     needs my Render login.
   - After I tell you the live onrender.com URL works, update the README's
     placeholder deploy section with a note that the app is live and add the
     URL near the top so my teammates can find it.

6. SUMMARY
   - When you're done, give me a short summary: which drive model you
     trained on, how many real failure records were in the training set, the
     real precision/recall/F1, the real concordance index from the survival
     model, and confirmation that all four pages passed the AppTest check
     with no exceptions.

Don't rewrite the dashboard design, the 3D components, the survival
analysis, the SHAP integration, the cost simulator, or the honesty
disclosures (the "what live means" banner, the early-warning-labeling
explanation, the "this is an adjustable assumption not a sourced statistic"
notes) — those are intentional, already reviewed, and are specifically what
make this project defensible as original work rather than a copy of a
standard Backblaze classifier tutorial. Only touch what's needed to get real
data flowing through the existing pipeline and ship it.
```

---

## After Antigravity finishes: pushing to GitHub yourself (if it can't)

Antigravity may not have GitHub credentials configured, in which case it'll
ask you to run the push yourself. From inside the project folder:

```bash
git init                                   # skip if already a repo
git add .
git commit -m "Real Backblaze data + deploy-ready build"
git branch -M main
git remote add origin https://github.com/<your-username>/hdd-failure-predictor.git
git push -u origin main
```

Create the empty repo at [github.com/new](https://github.com/new) first
(no README/license there — this project already has one). If Git asks for a
password, use a
[personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
instead of your account password, or sign in once via `gh auth login` if you
have the GitHub CLI.

## Deploying on Render yourself

1. Go to [render.com](https://render.com) and sign in with your GitHub
   account.
2. Click **New → Blueprint**, select the `hdd-failure-predictor` repo.
   Render reads the committed `render.yaml` and configures the build/start
   commands automatically — no manual form-filling.
3. Click **Apply** / **Create**. The first build takes a couple of minutes
   (installing pandas/scikit-learn/streamlit).
4. You'll get a live `https://hdd-failure-predictor.onrender.com`-style URL.
   Put this in your Assignment proposal, your presentation slides, and the
   email to your professor.
5. Every future `git push` to `main` auto-redeploys — so once you retrain on
   a different drive model or a newer quarter, just push and the live
   dashboard updates itself within a few minutes.

Remember: the free tier spins down after 15 minutes idle (~1 minute to wake
back up on the next visit) — mention this before a live demo so it doesn't
look broken. If that's a problem for a specific presentation, redeploy the
same repo to [Streamlit Community Cloud](https://streamlit.io/cloud) as a
backup link that doesn't sleep the same way.
