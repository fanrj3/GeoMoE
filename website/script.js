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
  const baseZooms = layers.map((layer) => Number(layer.dataset.baseZoom));
  const baseLogZooms = baseZooms.map((zoom) => Math.log2(zoom));
  const maxLogZoom = 6;
  const transitionWidth = 0.3;
  let frameRequested = false;

  function clamp(value, minimum = 0, maximum = 1) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function smoothstep(start, end, value) {
    const amount = clamp((value - start) / (end - start));
    return amount * amount * (3 - 2 * amount);
  }

  function renderMapZoom() {
    const bounds = mapZoom.getBoundingClientRect();
    const scrollDistance = Math.max(1, mapZoom.offsetHeight - window.innerHeight);
    const progress = clamp(-bounds.top / scrollDistance);
    const logZoom = progress * maxLogZoom;
    const globalZoom = 2 ** logZoom;
    const compactViewport = window.innerWidth <= 780;
    const mapViewport = mapZoom.querySelector(".geo-sticky");
    const viewportWidth = mapViewport.clientWidth;
    const viewportHeight = mapViewport.clientHeight;
    const renderedSize = Math.max(viewportWidth, viewportHeight);
    const renderedLeft = (viewportWidth - renderedSize) / 2;
    const renderedTop = (viewportHeight - renderedSize) / 2;
    const panProgress = smoothstep(0.03, 0.36, progress);
    const initialTargetX = viewportWidth * (compactViewport ? 0.08 : 0.18);
    const initialTargetY = viewportHeight * (compactViewport ? 0.5 : 0.46);
    const targetX = initialTargetX + (viewportWidth / 2 - initialTargetX) * panProgress;
    const targetY = initialTargetY + (viewportHeight / 2 - initialTargetY) * panProgress;
    const opacities = new Array(layers.length).fill(0);
    let activeIndex = 0;

    for (let index = 1; index < baseLogZooms.length; index += 1) {
      if (logZoom >= baseLogZooms[index]) activeIndex = index;
    }

    if (activeIndex === 0) {
      opacities[0] = 1;
    } else {
      const blend = smoothstep(
        baseLogZooms[activeIndex],
        baseLogZooms[activeIndex] + transitionWidth,
        logZoom,
      );
      opacities[activeIndex - 1] = 1 - blend;
      opacities[activeIndex] = blend;
      if (blend < 0.5) activeIndex -= 1;
    }

    layers.forEach((layer, index) => {
      const scale = globalZoom / baseZooms[index];
      const focusX = Number(layer.dataset.focusX);
      const focusY = Number(layer.dataset.focusY);
      const sourceTargetX = renderedLeft + renderedSize * focusX;
      const sourceTargetY = renderedTop + renderedSize * focusY;
      const translateX = targetX - sourceTargetX * scale;
      const translateY = targetY - sourceTargetY * scale;
      layer.style.opacity = String(opacities[index]);
      layer.style.transform = `matrix(${scale}, 0, 0, ${scale}, ${translateX}, ${translateY})`;
      layer.classList.toggle("is-active", index === activeIndex);
    });

    labels.forEach((label, index) => {
      label.classList.toggle("is-active", index === activeIndex);
    });

    if (progressBar) progressBar.style.transform = `scaleX(${progress})`;
    if (target) {
      target.style.left = `${targetX}px`;
      target.style.top = `${targetY}px`;
    }
    if (horizontalAxis) horizontalAxis.style.top = `${targetY}px`;
    if (verticalAxis) verticalAxis.style.left = `${targetX}px`;
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
