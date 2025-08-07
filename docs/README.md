# Portfolio Optimization - GitHub Pages

This folder contains the HTML version of the Portfolio Optimization notebook for deployment to GitHub Pages.

## Files

- `index.html` - The main HTML version of the Jupyter notebook
- `_config.yml` - GitHub Pages configuration file
- `notebook.html` - Previous version (if exists)

## Deployment Instructions

To deploy this to GitHub Pages:

1. **Enable GitHub Pages**:
   - Go to your repository settings on GitHub
   - Scroll down to "Pages" section
   - Under "Source", select "Deploy from a branch"
   - Choose "main" branch and "/docs" folder
   - Click "Save"

2. **Access your site**:
   - Your site will be available at: `https://yourusername.github.io/portfolio-optimization/`
   - It may take a few minutes to build and deploy

## Updating the HTML

To update the HTML version when you make changes to the notebook:

```bash
# Navigate to the project root
cd "c:\Prosjekter\Portfolio Optimization\portfolio-optimization"

# Convert notebook to HTML
python -m jupyter nbconvert --to html notebooks/notebook.ipynb --output-dir docs --output index.html
```

## Features

The HTML version includes:
- All markdown content with proper formatting
- Code cells with syntax highlighting
- All plots and visualizations (embedded as base64 images)
- Mathematical equations rendered with MathJax
- Responsive design that works on mobile devices
