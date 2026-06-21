# Personal website

A clean, minimal personal site built with [Jekyll](https://jekyllrb.com/) and hosted
on **GitHub Pages**. Three sections: **About**, **Projects**, and **Articles**.

GitHub builds the site for you — you do **not** need Ruby or any tooling installed to
publish. Just edit files and push.

---

## 1. Edit your content

Everything personal lives in a few obvious places:

| What | Where |
|------|-------|
| Name, tagline, email, social links, site URL | `_config.yml` |
| Your introduction (the About text) | `index.html` |
| Your projects | `_data/projects.yml` |
| Your articles | `_posts/` (one Markdown file per article) |

Open `_config.yml` first and confirm the values at the top are correct.

### Adding a project

Edit `_data/projects.yml` and add an entry:

```yaml
- name: "My Cool Project"
  repo: "my-cool-project"        # links to github.com/voidwalker-M/my-cool-project
  description: "A one-line description of what it does."
  tags: ["agents", "python"]
```

Use `url: "https://..."` instead of `repo:` to link somewhere other than your GitHub.

### Adding an article

Create a file in `_posts/` named `YYYY-MM-DD-title.md`, for example
`_posts/2026-07-01-planning-vs-reacting.md`:

```markdown
---
layout: post
title: "Planning vs. Reacting"
date: 2026-07-01
tags: [agents, planning]
---

Write your article in Markdown here...
```

Push it, and it appears under **Articles** automatically.

---

## 2. Publish to GitHub Pages

Your site will be live (for free) at **https://voidwalker-m.github.io**.

You already have a repo named `voidwalker-M.github.io`. It currently holds an old
2019 blog — these steps **replace** it with this new site. (No custom domain / no
`.com` needed — GitHub serves it on the free `github.io` address.)

### Steps

1. Push this `website/` folder to your repo, replacing the old content (commands below).
2. On GitHub, go to the repo → **Settings → Pages**:
   - **Build and deployment → Source: Deploy from a branch**, branch `master`,
     folder `/ (root)`. Save.
   - If a **Custom domain** is still set (e.g. `www.yaxinma.com`), **clear that field**
     so the site serves at `voidwalker-m.github.io`.
3. Wait ~1 minute, then visit **https://voidwalker-m.github.io**.

### Push commands

From inside the `website/` folder. This overwrites the old 2019 blog with a fresh
history (`-f`). The existing repo's default branch is `master`:

```bash
git init
git add .
git commit -m "New personal website"
git branch -M master
git remote add origin https://github.com/voidwalker-M/voidwalker-M.github.io.git
git push -f origin master
```

> The first push will ask for your GitHub credentials. Use a
> [Personal Access Token](https://github.com/settings/tokens) as the password
> (classic token with `repo` scope), not your account password.

---

## 3. (Optional) Preview locally

Only needed if you want to see changes before pushing. Requires Ruby.

```bash
bundle install
bundle exec jekyll serve
# open http://localhost:4000
```

---

## Project layout

```
website/
├── _config.yml          # site settings — edit your name, links, URL here
├── index.html           # About / home page
├── projects.html        # renders the Projects section
├── articles.html        # lists all articles
├── _data/projects.yml   # your project list
├── _posts/              # your articles (one Markdown file each)
├── _layouts/            # page templates (default + post)
└── assets/css/style.css # styling
```
