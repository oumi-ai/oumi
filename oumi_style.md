# Oumi-Inspired Style Guide

A comprehensive design system reference for building modern, tech-forward dark-mode interfaces.

---

## Design Philosophy

**Aesthetic Direction**: Professional Innovation
- Dark-first design that signals cutting-edge technology
- Gradient accents that create dynamism and forward momentum
- Clean, confident typography that builds trust
- Generous whitespace balanced with information density

**Core Principles**:
1. **Clarity over cleverness** - Every element serves a purpose
2. **Contrast creates hierarchy** - Use light on dark strategically
3. **Motion with meaning** - Animations enhance, never distract
4. **Consistency builds trust** - Systematic, predictable patterns

---

## Color System

### Primary Palette

```css
:root {
  /* Core Blues */
  --color-primary-deep: #062c82;
  --color-primary-accent: #3763ff;
  --color-primary-light: #5b7fff;

  /* Backgrounds */
  --color-bg-base: #0a0a0f;
  --color-bg-elevated: #12121a;
  --color-bg-card: #1a1a24;
  --color-bg-hover: #22222e;

  /* Text */
  --color-text-primary: #ffffff;
  --color-text-secondary: #73738c;
  --color-text-muted: #4a4a5c;

  /* Status */
  --color-success: #22c55e;
  --color-success-light: #4ade80;
  --color-error: #ef4444;
  --color-error-light: #f87171;

  /* Borders & Overlays */
  --color-border: rgba(255, 255, 255, 0.08);
  --color-border-hover: rgba(255, 255, 255, 0.15);
  --color-overlay: rgba(10, 10, 15, 0.8);
}
```

### Gradient Definitions

```css
:root {
  /* Hero/Feature Gradient */
  --gradient-primary: linear-gradient(100.89deg, #062c82 0%, #3763ff 100%);

  /* Subtle Background Gradient */
  --gradient-bg: linear-gradient(180deg, #0a0a0f 0%, #12121a 100%);

  /* Card Highlight */
  --gradient-card-border: linear-gradient(135deg, rgba(55, 99, 255, 0.3) 0%, rgba(6, 44, 130, 0.1) 100%);

  /* Text Gradient (for special headings) */
  --gradient-text: linear-gradient(90deg, #3763ff 0%, #5b7fff 100%);
}
```

---

## Typography

### Font Stack

```css
:root {
  /* Headings: Geometric, confident */
  --font-heading: 'Plus Jakarta Sans', 'Satoshi', system-ui, sans-serif;

  /* Body: Clean, readable */
  --font-body: 'Inter', 'SF Pro Text', system-ui, sans-serif;

  /* Mono: For code/technical */
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
}
```

### Type Scale

```css
:root {
  /* Headings */
  --text-6xl: clamp(3rem, 5vw, 4.5rem);      /* Hero titles */
  --text-5xl: clamp(2.5rem, 4vw, 3.5rem);    /* Section titles */
  --text-4xl: clamp(2rem, 3vw, 2.5rem);      /* Large headings */
  --text-3xl: clamp(1.5rem, 2.5vw, 2rem);    /* Medium headings */
  --text-2xl: clamp(1.25rem, 2vw, 1.5rem);   /* Small headings */
  --text-xl: 1.25rem;                         /* Large body */

  /* Body */
  --text-lg: 1.125rem;                        /* Primary body */
  --text-base: 1rem;                          /* Default body */
  --text-sm: 0.875rem;                        /* Secondary text */
  --text-xs: 0.75rem;                         /* Captions */

  /* Line Heights */
  --leading-tight: 1.1;
  --leading-snug: 1.25;
  --leading-normal: 1.5;
  --leading-relaxed: 1.625;

  /* Font Weights */
  --font-normal: 400;
  --font-medium: 500;
  --font-semibold: 600;
  --font-bold: 700;
}
```

### Typography Classes

```css
.heading-hero {
  font-family: var(--font-heading);
  font-size: var(--text-6xl);
  font-weight: var(--font-semibold);
  line-height: var(--leading-tight);
  letter-spacing: -0.02em;
  color: var(--color-text-primary);
}

.heading-section {
  font-family: var(--font-heading);
  font-size: var(--text-4xl);
  font-weight: var(--font-semibold);
  line-height: var(--leading-snug);
  letter-spacing: -0.01em;
  color: var(--color-text-primary);
}

.body-large {
  font-family: var(--font-body);
  font-size: var(--text-lg);
  font-weight: var(--font-normal);
  line-height: var(--leading-relaxed);
  color: var(--color-text-secondary);
}

.body-default {
  font-family: var(--font-body);
  font-size: var(--text-base);
  font-weight: var(--font-normal);
  line-height: var(--leading-normal);
  color: var(--color-text-secondary);
}
```

---

## Spacing & Layout

### Spacing Scale

```css
:root {
  --space-1: 0.25rem;   /* 4px */
  --space-2: 0.5rem;    /* 8px */
  --space-3: 0.75rem;   /* 12px */
  --space-4: 1rem;      /* 16px */
  --space-5: 1.25rem;   /* 20px */
  --space-6: 1.5rem;    /* 24px */
  --space-8: 2rem;      /* 32px */
  --space-10: 2.5rem;   /* 40px */
  --space-12: 3rem;     /* 48px */
  --space-16: 4rem;     /* 64px */
  --space-20: 5rem;     /* 80px */
  --space-24: 6rem;     /* 96px */
  --space-32: 8rem;     /* 128px */
}
```

### Container Widths

```css
:root {
  --container-sm: 640px;
  --container-md: 768px;
  --container-lg: 1024px;
  --container-xl: 1280px;
  --container-2xl: 1440px;
}

.container {
  width: 100%;
  max-width: var(--container-2xl);
  margin: 0 auto;
  padding: 0 var(--space-6);
}

@media (min-width: 768px) {
  .container { padding: 0 var(--space-8); }
}

@media (min-width: 1024px) {
  .container { padding: 0 var(--space-12); }
}
```

### Grid System

```css
.grid-2-col {
  display: grid;
  grid-template-columns: 1fr;
  gap: var(--space-6);
}

@media (min-width: 768px) {
  .grid-2-col {
    grid-template-columns: repeat(2, 1fr);
    gap: var(--space-8);
  }
}

.grid-3-col {
  display: grid;
  grid-template-columns: 1fr;
  gap: var(--space-6);
}

@media (min-width: 768px) {
  .grid-3-col { grid-template-columns: repeat(2, 1fr); }
}

@media (min-width: 1024px) {
  .grid-3-col { grid-template-columns: repeat(3, 1fr); }
}
```

---

## Components

### Buttons

```css
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
  padding: var(--space-3) var(--space-6);
  font-family: var(--font-body);
  font-size: var(--text-base);
  font-weight: var(--font-medium);
  line-height: 1;
  border-radius: 0.5rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn-primary {
  background: var(--color-primary-accent);
  color: var(--color-text-primary);
  border: none;
}

.btn-primary:hover {
  background: var(--color-primary-light);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(55, 99, 255, 0.4);
}

.btn-secondary {
  background: transparent;
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
}

.btn-secondary:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-border-hover);
}

.btn-lg {
  padding: var(--space-4) var(--space-8);
  font-size: var(--text-lg);
  border-radius: 0.75rem;
}
```

### Cards

```css
.card {
  background: var(--color-bg-card);
  border: 1px solid var(--color-border);
  border-radius: 1rem;
  padding: var(--space-8);
  transition: all 0.3s ease;
}

.card:hover {
  border-color: var(--color-border-hover);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
}

.card-elevated {
  background: var(--color-bg-elevated);
  box-shadow:
    0 4px 6px rgba(0, 0, 0, 0.1),
    0 10px 20px rgba(0, 0, 0, 0.15);
}

.card-gradient {
  position: relative;
  background: var(--color-bg-card);
  border: none;
}

.card-gradient::before {
  content: '';
  position: absolute;
  inset: 0;
  padding: 1px;
  border-radius: 1rem;
  background: var(--gradient-card-border);
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
}
```

### Icon Containers

```css
.icon-circle {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2.5rem;
  height: 2.5rem;
  border-radius: 50%;
  background: rgba(55, 99, 255, 0.15);
  color: var(--color-primary-accent);
}

.icon-circle-success {
  background: rgba(34, 197, 94, 0.15);
  color: var(--color-success);
}

.icon-circle-error {
  background: rgba(239, 68, 68, 0.15);
  color: var(--color-error);
}
```

### Badges

```css
.badge {
  display: inline-flex;
  align-items: center;
  padding: var(--space-1) var(--space-3);
  font-size: var(--text-xs);
  font-weight: var(--font-medium);
  border-radius: 9999px;
  background: rgba(55, 99, 255, 0.15);
  color: var(--color-primary-accent);
}
```

---

## Visual Effects

### Shadows

```css
:root {
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.2);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.2), 0 2px 4px rgba(0, 0, 0, 0.1);
  --shadow-lg: 0 10px 20px rgba(0, 0, 0, 0.25), 0 6px 6px rgba(0, 0, 0, 0.15);
  --shadow-xl: 0 20px 40px rgba(0, 0, 0, 0.3), 0 10px 10px rgba(0, 0, 0, 0.2);
  --shadow-glow: 0 0 40px rgba(55, 99, 255, 0.3);
}
```

### Animations

```css
/* Fade In Up */
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.animate-fade-in-up {
  animation: fadeInUp 0.6s ease-out forwards;
}

/* Staggered children */
.stagger-children > * {
  opacity: 0;
  animation: fadeInUp 0.5s ease-out forwards;
}

.stagger-children > *:nth-child(1) { animation-delay: 0.1s; }
.stagger-children > *:nth-child(2) { animation-delay: 0.2s; }
.stagger-children > *:nth-child(3) { animation-delay: 0.3s; }
.stagger-children > *:nth-child(4) { animation-delay: 0.4s; }

/* Grayscale hover transition */
.grayscale-hover {
  filter: grayscale(100%);
  opacity: 0.6;
  transition: all 0.3s ease;
}

.grayscale-hover:hover {
  filter: grayscale(0%);
  opacity: 1;
}

/* Subtle float */
@keyframes float {
  0%, 100% { transform: translateY(0) rotate(-12deg); }
  50% { transform: translateY(-10px) rotate(-12deg); }
}

.animate-float {
  animation: float 6s ease-in-out infinite;
}
```

---

## Section Patterns

### Hero Section

```css
.hero {
  position: relative;
  min-height: 90vh;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  overflow: hidden;
}

.hero-bg-video {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  opacity: 0.4;
}

.hero-overlay {
  position: absolute;
  inset: 0;
  background: linear-gradient(180deg, transparent 0%, var(--color-bg-base) 100%);
}

.hero-content {
  position: relative;
  z-index: 10;
  max-width: 800px;
  padding: var(--space-8);
}
```

### Comparison Section

```css
.comparison-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: var(--space-6);
}

@media (min-width: 768px) {
  .comparison-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

.comparison-old {
  background: var(--color-bg-elevated);
  border-color: var(--color-border);
}

.comparison-new {
  background: var(--gradient-primary);
  border: none;
}

.comparison-item {
  display: flex;
  align-items: flex-start;
  gap: var(--space-3);
  padding: var(--space-4) 0;
}

.comparison-item + .comparison-item {
  border-top: 1px solid var(--color-border);
}
```

### Logo Wall

```css
.logo-wall {
  padding: var(--space-16) 0;
  background: var(--gradient-primary);
}

.logo-grid {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: var(--space-8);
}

.logo-item {
  height: 2.5rem;
  filter: grayscale(100%) brightness(2);
  opacity: 0.7;
  transition: all 0.3s ease;
}

.logo-item:hover {
  filter: grayscale(0%) brightness(1);
  opacity: 1;
}
```

---

## Usage as a Prompt

When building a new website with this style, use this prompt:

> **Build a [type of website] using the Oumi design aesthetic:**
>
> - **Theme**: Dark-mode first with deep navy (#062c82) and bright blue (#3763ff) accents
> - **Background**: Near-black (#0a0a0f) with subtle elevation layers
> - **Typography**: Geometric sans-serif headings (Plus Jakarta Sans/Satoshi), clean body text (Inter), semibold weights for emphasis
> - **Cards**: Rounded corners (1rem), semi-transparent borders, elevated shadows on hover
> - **Buttons**: Solid blue primary with glow on hover, outlined secondary
> - **Effects**: Linear gradients (100deg angle), grayscale-to-color logo hovers, staggered fade-in animations
> - **Layout**: Max-width 1440px, generous padding, 2-column responsive grids
> - **Mood**: Professional, innovative, tech-forward, trustworthy
>
> Include comparison sections with contrasting backgrounds, icon circles for feature lists, and badge pills for highlighting key terms.

---

This style guide gives you everything needed to recreate the Oumi aesthetic consistently across different projects. The CSS variables and component patterns can be directly copied and customized for your specific needs.
