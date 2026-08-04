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
  const mapViewport = mapZoom.querySelector(".geo-sticky");
  const mapCanvas = mapZoom.querySelector("[data-map-canvas]");
  const mapContext = mapCanvas?.getContext("2d", { alpha: false });
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
  const layerReady = new Array(layers.length).fill(false);
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

  function markLayerReady(layer, index) {
    const finish = () => {
      if (!layer.naturalWidth) return;
      layerReady[index] = true;
      requestMapFrame();
    };

    if (typeof layer.decode === "function") {
      layer.decode().then(finish).catch(finish);
    } else {
      finish();
    }
  }

  function sizeCanvas(width, height) {
    if (!mapCanvas || !mapContext) return;
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
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

      const sourceX = (destinationLeft - drawLeft) / drawSize * layer.naturalWidth;
      const sourceY = (destinationTop - drawTop) / drawSize * layer.naturalHeight;
      const sourceWidth = destinationWidth / drawSize * layer.naturalWidth;
      const sourceHeight = destinationHeight / drawSize * layer.naturalHeight;

      mapContext.globalAlpha = visibleLayerIndex === 0 ? 1 : opacity;
      mapContext.drawImage(
        layer,
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
      markLayerReady(layer, index);
    } else {
      layer.addEventListener("load", () => markLayerReady(layer, index), { once: true });
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
      Math.min(window.devicePixelRatio || 1, 1.5),
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

    if (activeIndex === 0) {
      idealOpacities[0] = 1;
    } else {
      const blend = smoothstep(
        baseLogZooms[activeIndex],
        baseLogZooms[activeIndex] + transitionWidth,
        logZoom,
      );
      idealOpacities[activeIndex - 1] = 1 - blend;
      idealOpacities[activeIndex] = blend;
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
  renderMapZoom();
}
