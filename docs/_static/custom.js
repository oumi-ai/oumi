/**
 * OUMI Documentation - Custom JavaScript
 * Enhanced interactions and visual effects
 */

(function() {
    'use strict';

    // Wait for DOM to be ready
    document.addEventListener('DOMContentLoaded', function() {
        initScrollProgress();
        initSmoothScrolling();
        initCodeBlockEnhancements();
        initTableOfContentsHighlight();
        initSearchEnhancements();
        initKeyboardNavigation();
        initAnimations();
    });

    /**
     * Reading Progress Indicator
     * Shows a progress bar at the top of the page
     */
    function initScrollProgress() {
        // Create progress bar element
        const progressBar = document.createElement('div');
        progressBar.id = 'oumi-scroll-progress';
        progressBar.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 0%;
            height: 3px;
            background: linear-gradient(90deg, #00D4AA, #00A888);
            z-index: 9999;
            transition: width 50ms ease-out;
            box-shadow: 0 0 10px rgba(0, 212, 170, 0.5);
        `;
        document.body.appendChild(progressBar);

        // Update progress on scroll
        function updateProgress() {
            const scrollTop = window.scrollY;
            const docHeight = document.documentElement.scrollHeight - window.innerHeight;
            const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
            progressBar.style.width = progress + '%';
        }

        window.addEventListener('scroll', updateProgress, { passive: true });
        updateProgress();
    }

    /**
     * Smooth Scrolling for Anchor Links
     */
    function initSmoothScrolling() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                const href = this.getAttribute('href');
                if (href === '#') return;

                const target = document.querySelector(href);
                if (target) {
                    e.preventDefault();
                    const headerOffset = 80;
                    const elementPosition = target.getBoundingClientRect().top;
                    const offsetPosition = elementPosition + window.scrollY - headerOffset;

                    window.scrollTo({
                        top: offsetPosition,
                        behavior: 'smooth'
                    });

                    // Update URL without triggering navigation
                    history.pushState(null, null, href);
                }
            });
        });
    }

    /**
     * Code Block Enhancements
     * - Language label display
     */
    function initCodeBlockEnhancements() {
        // Add language labels to code blocks
        document.querySelectorAll('div[class*="highlight-"]').forEach(block => {
            const classes = block.className.split(' ');
            const langClass = classes.find(c => c.startsWith('highlight-'));
            if (langClass) {
                const lang = langClass.replace('highlight-', '').toUpperCase();
                if (lang && lang !== 'DEFAULT') {
                    const label = document.createElement('span');
                    label.className = 'oumi-code-lang';
                    label.textContent = lang;
                    label.style.cssText = `
                        position: absolute;
                        top: 8px;
                        right: 8px;
                        font-family: 'JetBrains Mono', monospace;
                        font-size: 0.65rem;
                        font-weight: 600;
                        text-transform: uppercase;
                        letter-spacing: 0.05em;
                        color: var(--oumi-text-muted, #6E7681);
                        background: var(--oumi-bg-elevated, #242B35);
                        padding: 4px 8px;
                        border-radius: 4px;
                        opacity: 0.8;
                        pointer-events: none;
                    `;
                    block.style.position = 'relative';
                    block.appendChild(label);
                }
            }
        });

        // Add click-to-copy feedback using safe DOM methods
        document.querySelectorAll('button.copybtn').forEach(btn => {
            btn.addEventListener('click', function() {
                const originalBg = this.style.backgroundColor;
                const originalBorder = this.style.borderColor;

                // Show copied state with color change
                this.style.backgroundColor = 'var(--oumi-success, #3FB950)';
                this.style.borderColor = 'var(--oumi-success, #3FB950)';

                setTimeout(() => {
                    this.style.backgroundColor = originalBg;
                    this.style.borderColor = originalBorder;
                }, 2000);
            });
        });
    }

    /**
     * Table of Contents - Active Section Highlighting
     */
    function initTableOfContentsHighlight() {
        const tocLinks = document.querySelectorAll('.bd-toc .nav-link');
        if (tocLinks.length === 0) return;

        const sections = [];
        tocLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href && href.startsWith('#')) {
                const section = document.querySelector(href);
                if (section) {
                    sections.push({ link, section });
                }
            }
        });

        function highlightActiveSection() {
            const scrollPos = window.scrollY + 100;

            let activeSection = null;
            sections.forEach(({ link, section }) => {
                if (section.offsetTop <= scrollPos) {
                    activeSection = link;
                }
            });

            tocLinks.forEach(link => {
                link.classList.remove('toc-active');
                link.style.color = '';
                link.style.fontWeight = '';
            });

            if (activeSection) {
                activeSection.classList.add('toc-active');
                activeSection.style.color = 'var(--oumi-accent, #00D4AA)';
                activeSection.style.fontWeight = '600';
            }
        }

        window.addEventListener('scroll', highlightActiveSection, { passive: true });
        highlightActiveSection();
    }

    /**
     * Search Enhancements
     */
    function initSearchEnhancements() {
        // Add keyboard shortcut hint to search
        const searchInput = document.querySelector('.bd-search input, input[type="search"]');
        if (searchInput) {
            // Create shortcut hint
            const hint = document.createElement('kbd');
            hint.textContent = '/';
            hint.style.cssText = `
                position: absolute;
                right: 12px;
                top: 50%;
                transform: translateY(-50%);
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.75rem;
                padding: 2px 6px;
                background: var(--oumi-bg-elevated, #242B35);
                border: 1px solid var(--oumi-border, #30363D);
                border-radius: 4px;
                color: var(--oumi-text-muted, #6E7681);
                pointer-events: none;
            `;

            const searchContainer = searchInput.closest('.bd-search');
            if (searchContainer) {
                searchContainer.style.position = 'relative';
                searchContainer.appendChild(hint);

                // Hide hint when input is focused
                searchInput.addEventListener('focus', () => hint.style.display = 'none');
                searchInput.addEventListener('blur', () => {
                    if (!searchInput.value) hint.style.display = '';
                });
            }
        }
    }

    /**
     * Keyboard Navigation
     */
    function initKeyboardNavigation() {
        document.addEventListener('keydown', function(e) {
            // Skip if user is typing in an input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                // ESC to blur input
                if (e.key === 'Escape') {
                    e.target.blur();
                }
                return;
            }

            // "/" to focus search
            if (e.key === '/') {
                const searchInput = document.querySelector('.bd-search input, input[type="search"]');
                if (searchInput) {
                    e.preventDefault();
                    searchInput.focus();
                }
            }

            // "t" to scroll to top
            if (e.key === 't' && !e.ctrlKey && !e.metaKey) {
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }

            // "g" + "h" for home (gg style)
            if (e.key === 'g') {
                const waitForH = function(e2) {
                    if (e2.key === 'h') {
                        window.location.href = '/';
                    }
                    document.removeEventListener('keydown', waitForH);
                };
                setTimeout(() => document.removeEventListener('keydown', waitForH), 1000);
                document.addEventListener('keydown', waitForH);
            }
        });
    }

    /**
     * Animation Enhancements
     * Intersection Observer for scroll-triggered animations
     */
    function initAnimations() {
        // Check for reduced motion preference
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
            return;
        }

        // Animate elements when they come into view
        const animatableElements = document.querySelectorAll(
            '.sd-card, .admonition, table, dl.py, .cell'
        );

        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                    observer.unobserve(entry.target);
                }
            });
        }, observerOptions);

        animatableElements.forEach(el => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(20px)';
            el.style.transition = 'opacity 0.5s ease-out, transform 0.5s ease-out';
            observer.observe(el);
        });

        // Add hover effect to navigation items
        document.querySelectorAll('.bd-sidebar-primary .nav-link').forEach(link => {
            link.addEventListener('mouseenter', function() {
                this.style.transform = 'translateX(4px)';
            });
            link.addEventListener('mouseleave', function() {
                this.style.transform = '';
            });
        });
    }

    /**
     * Dark/Light Mode Toggle Enhancement
     * Smooth transition when switching themes
     */
    function initThemeTransition() {
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.attributeName === 'data-theme') {
                    document.body.style.transition = 'background-color 0.3s ease, color 0.3s ease';
                    setTimeout(() => {
                        document.body.style.transition = '';
                    }, 300);
                }
            });
        });

        observer.observe(document.documentElement, { attributes: true });
    }

    // Initialize theme transition on load
    initThemeTransition();

})();
