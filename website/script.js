const menuButton = document.querySelector(".menu-button");
const navigation = document.querySelector(".nav-links");

if (menuButton && navigation) {
  menuButton.addEventListener("click", () => {
    const isOpen = menuButton.getAttribute("aria-expanded") === "true";
    menuButton.setAttribute("aria-expanded", String(!isOpen));
    menuButton.setAttribute("aria-label", isOpen ? "Open navigation" : "Close navigation");
    navigation.dataset.open = String(!isOpen);
  });

  navigation.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      menuButton.setAttribute("aria-expanded", "false");
      menuButton.setAttribute("aria-label", "Open navigation");
      navigation.dataset.open = "false";
    });
  });
}

const copyButton = document.querySelector("[data-copy-bibtex]");
const bibtex = document.querySelector("#bibtex code");

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch {
      // Fall through for browsers that expose Clipboard API without permission.
    }
  }

  const textArea = document.createElement("textarea");
  textArea.value = text;
  textArea.style.position = "fixed";
  textArea.style.opacity = "0";
  document.body.appendChild(textArea);
  textArea.focus();
  textArea.select();
  const copied = document.execCommand("copy");
  textArea.remove();

  if (!copied) {
    throw new Error("Copy command was rejected");
  }
}

if (copyButton && bibtex) {
  copyButton.addEventListener("click", async () => {
    const originalLabel = copyButton.textContent;

    try {
      await copyText(bibtex.textContent.trim());
      copyButton.textContent = "Copied";
    } catch {
      copyButton.textContent = "Copy failed";
    }

    window.setTimeout(() => {
      copyButton.textContent = originalLabel;
    }, 1800);
  });
}

const mapZoom = document.querySelector("[data-map-zoom]");

if (mapZoom) {
  const layers = [...mapZoom.querySelectorAll("[data-map-layer]")];
  const labels = [...mapZoom.querySelectorAll("[data-map-label]")];
  const progressBar = mapZoom.querySelector("[data-map-progress]");
  const target = mapZoom.querySelector(".geo-target");
  const horizontalAxis = mapZoom.querySelector(".geo-axis-x");
  const verticalAxis = mapZoom.querySelector(".geo-axis-y");
  const stageName = mapZoom.querySelector("[data-map-stage]");
  const stageResolution = mapZoom.querySelector("[data-map-resolution]");
  const stages = [
    { name: "City gallery", resolution: "49,152 px source" },
    { name: "Regional candidate", resolution: "12,288 px window" },
    { name: "District candidate", resolution: "4,096 px window" },
    { name: "Local candidate", resolution: "2 x 2 L3 tiles" },
  ];
  let frameRequested = false;

  function renderMapZoom() {
    const bounds = mapZoom.getBoundingClientRect();
    const scrollDistance = Math.max(1, mapZoom.offsetHeight - window.innerHeight);
    const progress = Math.min(1, Math.max(0, -bounds.top / scrollDistance));
    const phase = progress * (layers.length - 1);
    const activeIndex = Math.round(phase);
    const firstLevelProgress = Math.min(1, phase);
    const compactViewport = window.innerWidth <= 780;
    const targetX = compactViewport ? 50 : 18 + firstLevelProgress * 32;
    const targetY = compactViewport ? 50 : 46 + firstLevelProgress * 4;

    layers.forEach((layer, index) => {
      const opacity = Math.max(0, 1 - Math.abs(index - phase));
      const localProgress = Math.min(1, Math.max(0, phase - index));
      const scale = 1 + localProgress * (index === 0 ? 0.22 : 0.08);
      layer.style.opacity = String(opacity);
      layer.style.transform = `scale(${scale})`;
      layer.classList.toggle("is-active", opacity > 0.5);
    });

    labels.forEach((label, index) => {
      label.classList.toggle("is-active", index === activeIndex);
    });

    if (progressBar) progressBar.style.transform = `scaleX(${progress})`;
    if (target) {
      target.style.left = `${targetX}%`;
      target.style.top = `${targetY}%`;
    }
    if (horizontalAxis) horizontalAxis.style.top = `${targetY}%`;
    if (verticalAxis) verticalAxis.style.left = `${targetX}%`;
    if (stageName) stageName.textContent = stages[activeIndex].name;
    if (stageResolution) stageResolution.textContent = stages[activeIndex].resolution;
    frameRequested = false;
  }

  function requestMapFrame() {
    if (frameRequested) return;
    frameRequested = true;
    window.requestAnimationFrame(renderMapZoom);
  }

  window.addEventListener("scroll", requestMapFrame, { passive: true });
  window.addEventListener("resize", requestMapFrame);
  renderMapZoom();
}
