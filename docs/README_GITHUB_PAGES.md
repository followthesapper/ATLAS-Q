# GitHub Pages Setup for ATLAS-Q Documentation

This directory is configured to serve as a GitHub Pages documentation site.

---

## 🌐 Live Site

Once enabled, the documentation will be available at:
**https://followthsapper.github.io/ATLAS-Q/**

---

## 🔧 Enabling GitHub Pages

### One-Time Setup (Repository Owner)

1. Go to repository **Settings** → **Pages**
2. Under **Build and deployment**:
   - **Source:** Select "GitHub Actions"
   - (The workflow file `.github/workflows/pages.yml` will handle the rest)
3. Save changes
4. Push any commit to trigger the first deployment
5. Site will be live at `https://followthsapper.github.io/ATLAS-Q/` in ~2-5 minutes

---

## 📁 Site Structure

```
docs/
├── index.md                  # Homepage
├── _config.yml               # Jekyll configuration
├── COMPLETE_GUIDE.md         # Full guide
├── FEATURE_STATUS.md         # Implementation status
├── WHITEPAPER.md             # Technical architecture
├── RESEARCH_PAPER.md         # Math & algorithms
├── OVERVIEW.md               # High-level overview
└── README_GITHUB_PAGES.md    # This file
```

---

## 🎨 Theme

The site uses the **Cayman** theme, which provides:
- Clean, modern design
- Responsive layout (mobile-friendly)
- Syntax highlighting for code blocks
- GitHub-style markdown rendering

---

## 📝 Adding New Documentation

1. Create a new `.md` file in `docs/`
2. Add front matter (optional):
   ```yaml
   ---
   title: Your Page Title
   description: Brief description
   ---
   ```
3. Write content using GitHub-flavored Markdown
4. Add link to navigation in `_config.yml` if needed
5. Commit and push - site auto-updates!

---

## 🔍 Testing Locally

To test the site locally before pushing:

```bash
# Install Jekyll
gem install bundler jekyll

# Create Gemfile
cd docs
cat > Gemfile << 'EOF'
source "https://rubygems.org"
gem "github-pages", group: :jekyll_plugins
EOF

# Install dependencies
bundle install

# Serve locally
bundle exec jekyll serve

# Open http://localhost:4000/ATLAS-Q/
```

---

## 🚀 Automatic Deployment

The site automatically rebuilds when:
- Any file in `docs/` is modified
- `.github/workflows/pages.yml` is updated
- Manual trigger via GitHub Actions UI

**Deployment time:** ~2-5 minutes after push

---

## 🎯 Best Practices

### ✅ DO:
- Use relative links: `[Text](COMPLETE_GUIDE)` (no `.md` extension)
- Keep markdown simple (GitHub-flavored)
- Test locally before pushing major changes
- Use code blocks with language tags: ` ```python `

### ❌ DON'T:
- Use absolute paths like `/docs/...`
- Include HTML unless necessary
- Use complex Jekyll features (keep it simple)
- Link to files not in `docs/` folder

---

## 🐛 Troubleshooting

### Issue: Page not found (404)
- Check file exists in `docs/` folder
- Verify link uses relative path without `.md`
- Wait 5 minutes for cache to clear

### Issue: Build failing
- Check GitHub Actions tab for errors
- Verify `_config.yml` syntax is valid
- Ensure all linked files exist

### Issue: Formatting looks wrong
- Verify markdown syntax
- Check code blocks have proper fencing
- Test locally with Jekyll

---

## 📚 Additional Resources

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [Cayman Theme](https://github.com/pages-themes/cayman)

---

**Last Updated:** October 2025
