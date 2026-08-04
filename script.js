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
  const jumpButtons = [...mapZoom.querySelectorAll("[data-map-jump]")];
  const progressBar = mapZoom.querySelector("[data-map-progress]");
  const target = mapZoom.querySelector(".geo-target");
  const horizontalAxis = mapZoom.querySelector(".geo-axis-x");
  const verticalAxis = mapZoom.querySelector(".geo-axis-y");
  const stageName = mapZoom.querySelector("[data-map-stage]");
  const stageResolution = mapZoom.querySelector("[data-map-resolution]");
  const mapViewport = mapZoom.querySelector(".geo-sticky");
  const mapCanvas = mapZoom.querySelector("[data-map-canvas]");
  const mapContext = mapCanvas?.getContext("2d", { alpha: false });
  const stages = [
    { name: "City-scale gallery", resolution: "49,152 px source" },
    { name: "Regional candidate", resolution: "12,288 px window" },
    { name: "District candidate", resolution: "4,096 px window" },
    { name: "Local candidate", resolution: "2 x 2 L3 tiles" },
  ];
  const baseZooms = layers.map((layer) => Number(layer.dataset.baseZoom));
  const baseLogZooms = baseZooms.map((zoom) => Math.log2(zoom));
  const maxLogZoom = 6;
  const transitionWidth = 0.46;
  const layerReady = new Array(layers.length).fill(false);
  const layerSources = new Array(layers.length).fill(null);
  let frameRequested = false;
  let lastRenderSignature = "";
  let renderedStageIndex = -1;

  function clamp(value, minimum = 0, maximum = 1) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function smoothstep(start, end, value) {
    const amount = clamp((value - start) / (end - start));
    return amount * amount * (3 - 2 * amount);
  }

  const warmCanvas = document.createElement("canvas");
  warmCanvas.width = 1;
  warmCanvas.height = 1;
  const warmContext = warmCanvas.getContext("2d");

  async function prepareLayer(layer, index) {
    if (layerReady[index] || !layer.naturalWidth) return;

    if (typeof layer.decode === "function") {
      try {
        await layer.decode();
      } catch {
        // A loaded image remains usable if explicit decoding is interrupted.
      }
    }

    let source = layer;
    if ("createImageBitmap" in window) {
      try {
        source = await createImageBitmap(layer);
      } catch {
        source = layer;
      }
    }

    layerSources[index] = source;
    if (warmContext) {
      warmContext.clearRect(0, 0, 1, 1);
      warmContext.drawImage(source, 0, 0, 1, 1);
    }
    layerReady[index] = true;
    lastRenderSignature = "";
    requestMapFrame();
  }

  function sizeCanvas(width, height) {
    if (!mapCanvas || !mapContext) return;
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    const outputWidth = Math.round(width * pixelRatio);
    const outputHeight = Math.round(height * pixelRatio);

    if (mapCanvas.width !== outputWidth || mapCanvas.height !== outputHeight) {
      mapCanvas.width = outputWidth;
      mapCanvas.height = outputHeight;
      mapCanvas.style.width = `${width}px`;
      mapCanvas.style.height = `${height}px`;
    }

    mapContext.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    mapContext.imageSmoothingEnabled = true;
    mapContext.imageSmoothingQuality = "high";
  }

  function drawVisibleMap({
    globalZoom,
    opacities,
    renderedSize,
    targetX,
    targetY,
    viewportWidth,
    viewportHeight,
  }) {
    if (!mapCanvas || !mapContext) return;
    const visibleLayers = opacities
      .map((opacity, index) => ({ index, opacity }))
      .filter(({ opacity, index }) => opacity > 0.001 && layerReady[index]);

    if (!visibleLayers.length) return;

    sizeCanvas(viewportWidth, viewportHeight);
    mapContext.globalAlpha = 1;
    mapContext.fillStyle = "#07110f";
    mapContext.fillRect(0, 0, viewportWidth, viewportHeight);

    visibleLayers.forEach(({ index, opacity }, visibleLayerIndex) => {
      const layer = layers[index];
      const source = layerSources[index] || layer;
      const sourceNaturalWidth = source.width || layer.naturalWidth;
      const sourceNaturalHeight = source.height || layer.naturalHeight;
      const scale = globalZoom / baseZooms[index];
      const drawSize = renderedSize * scale;
      const focusX = Number(layer.dataset.focusX);
      const focusY = Number(layer.dataset.focusY);
      const drawLeft = targetX - drawSize * focusX;
      const drawTop = targetY - drawSize * focusY;
      const destinationLeft = Math.max(0, drawLeft);
      const destinationTop = Math.max(0, drawTop);
      const destinationRight = Math.min(viewportWidth, drawLeft + drawSize);
      const destinationBottom = Math.min(viewportHeight, drawTop + drawSize);
      const destinationWidth = destinationRight - destinationLeft;
      const destinationHeight = destinationBottom - destinationTop;

      if (destinationWidth <= 0 || destinationHeight <= 0) return;

      const sourceX = (destinationLeft - drawLeft) / drawSize * sourceNaturalWidth;
      const sourceY = (destinationTop - drawTop) / drawSize * sourceNaturalHeight;
      const sourceWidth = destinationWidth / drawSize * sourceNaturalWidth;
      const sourceHeight = destinationHeight / drawSize * sourceNaturalHeight;

      mapContext.globalAlpha = visibleLayerIndex === 0 ? 1 : opacity;
      mapContext.drawImage(
        source,
        sourceX,
        sourceY,
        sourceWidth,
        sourceHeight,
        destinationLeft,
        destinationTop,
        destinationWidth,
        destinationHeight,
      );
    });

    mapContext.globalAlpha = 1;
    mapCanvas.classList.add("is-ready");
  }

  layers.forEach((layer, index) => {
    if (layer.complete && layer.naturalWidth) {
      prepareLayer(layer, index);
    } else {
      layer.addEventListener("load", () => prepareLayer(layer, index), { once: true });
    }
  });

  function renderMapZoom() {
    const bounds = mapZoom.getBoundingClientRect();
    const scrollDistance = Math.max(1, mapZoom.offsetHeight - window.innerHeight);
    const progress = clamp(-bounds.top / scrollDistance);
    const logZoom = progress * maxLogZoom;
    const globalZoom = 2 ** logZoom;
    const compactViewport = window.innerWidth <= 780;
    const viewportWidth = mapViewport.clientWidth;
    const viewportHeight = mapViewport.clientHeight;
    const renderSignature = [
      progress.toFixed(6),
      viewportWidth,
      viewportHeight,
      Math.min(window.devicePixelRatio || 1, 2),
      layerReady.map(Number).join(""),
    ].join(":");

    if (renderSignature === lastRenderSignature) {
      frameRequested = false;
      return;
    }
    lastRenderSignature = renderSignature;

    const renderedSize = Math.max(viewportWidth, viewportHeight);
    const renderedLeft = (viewportWidth - renderedSize) / 2;
    const renderedTop = (viewportHeight - renderedSize) / 2;
    const panProgress = smoothstep(0.03, 0.36, progress);
    const naturalTargetX = renderedLeft + renderedSize * Number(layers[0].dataset.focusX);
    const naturalTargetY = renderedTop + renderedSize * Number(layers[0].dataset.focusY);
    const initialTargetX = compactViewport
      ? Math.max(viewportWidth * 0.08, naturalTargetX)
      : naturalTargetX;
    const initialTargetY = clamp(naturalTargetY, viewportHeight * 0.08, viewportHeight * 0.92);
    const targetX = initialTargetX + (viewportWidth / 2 - initialTargetX) * panProgress;
    const targetY = initialTargetY + (viewportHeight / 2 - initialTargetY) * panProgress;
    const idealOpacities = new Array(layers.length).fill(0);
    const opacities = new Array(layers.length).fill(0);
    let activeIndex = 0;

    for (let index = 1; index < baseLogZooms.length; index += 1) {
      if (logZoom >= baseLogZooms[index]) activeIndex = index;
    }

    const nextIndex = activeIndex + 1;
    if (nextIndex < layers.length) {
      const transitionEnd = baseLogZooms[nextIndex];
      const transitionStart = transitionEnd - transitionWidth;
      const blend = smoothstep(transitionStart, transitionEnd, logZoom);
      idealOpacities[activeIndex] = 1 - blend;
      idealOpacities[nextIndex] = blend;
    } else {
      idealOpacities[activeIndex] = 1;
    }

    const readyOpacity = idealOpacities.reduce(
      (total, opacity, index) => total + (layerReady[index] ? opacity : 0),
      0,
    );

    if (readyOpacity > 0.001) {
      idealOpacities.forEach((opacity, index) => {
        if (layerReady[index]) opacities[index] = opacity / readyOpacity;
      });
    } else {
      let fallbackIndex = activeIndex;
      while (fallbackIndex >= 0 && !layerReady[fallbackIndex]) fallbackIndex -= 1;
      if (fallbackIndex < 0) fallbackIndex = layerReady.findIndex(Boolean);
      if (fallbackIndex >= 0) opacities[fallbackIndex] = 1;
    }

    const visibleIndex = opacities.reduce(
      (bestIndex, opacity, index) => opacity > opacities[bestIndex] ? index : bestIndex,
      0,
    );

    drawVisibleMap({
      globalZoom,
      opacities,
      renderedSize,
      targetX,
      targetY,
      viewportWidth,
      viewportHeight,
    });

    if (visibleIndex !== renderedStageIndex) {
      labels.forEach((label, index) => {
        label.classList.toggle("is-active", index === visibleIndex);
        const button = label.querySelector("[data-map-jump]");
        if (button) {
          if (index === visibleIndex) button.setAttribute("aria-current", "step");
          else button.removeAttribute("aria-current");
        }
      });
      if (stageName) stageName.textContent = stages[visibleIndex].name;
      if (stageResolution) stageResolution.textContent = stages[visibleIndex].resolution;
      renderedStageIndex = visibleIndex;
    }

    if (progressBar) progressBar.style.transform = `scaleX(${progress})`;
    if (target) {
      target.style.left = `${targetX}px`;
      target.style.top = `${targetY}px`;
    }
    if (horizontalAxis) horizontalAxis.style.top = `${targetY}px`;
    if (verticalAxis) verticalAxis.style.left = `${targetX}px`;
    frameRequested = false;
  }

  function requestMapFrame() {
    if (frameRequested) return;
    frameRequested = true;
    window.requestAnimationFrame(renderMapZoom);
  }

  window.addEventListener("scroll", requestMapFrame, { passive: true });
  window.addEventListener("resize", requestMapFrame);

  jumpButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const level = Number(button.dataset.mapLevel);
      if (!Number.isInteger(level) || level < 0 || level >= baseLogZooms.length) return;

      const sectionTop = window.scrollY + mapZoom.getBoundingClientRect().top;
      const scrollDistance = Math.max(mapZoom.offsetHeight - window.innerHeight, 1);
      const levelProgress = baseLogZooms[level] / maxLogZoom;
      const targetTop = sectionTop + levelProgress * scrollDistance;
      const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      window.scrollTo({ top: targetTop, behavior: reducedMotion ? "auto" : "smooth" });
    });
  });

  renderMapZoom();
}
