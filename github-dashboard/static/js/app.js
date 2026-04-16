// HTMX global config
htmx.config.defaultSwapStyle = "innerHTML";
htmx.config.defaultSwapDelay = 0;
htmx.config.defaultSettleDelay = 100;

// Show rate limit warning when triggered by server
document.addEventListener("htmx:afterOnLoad", function (evt) {
  const triggerHeader = evt.detail.xhr.getResponseHeader("HX-Trigger");
  if (triggerHeader && triggerHeader.includes("rateLimitExceeded")) {
    const badge = document.getElementById("rate-limit-badge");
    if (badge) badge.classList.remove("hidden");
  }
});
