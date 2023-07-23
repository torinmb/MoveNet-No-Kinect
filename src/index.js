/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import 'regenerator-runtime/runtime'

import '@tensorflow/tfjs-backend-webgl';
import * as mpPose from '@mediapipe/pose';

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

// tfjsWasm.setWasmPaths(
//     `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
//         tfjsWasm.version_wasm}/dist/`);

import * as posedetection from '@tensorflow-models/pose-detection';

import {Camera} from './camera';
import {setupDatGui} from './option_panel';
import {STATE, BLAZEPOSE_CONFIG, MOVENET_CONFIG} from './params';
import {setupStats} from './stats_panel';
import {setBackendAndEnvFlags} from './util';

let detector, camera, stats;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let rafId;
let webcamDevices;
let ws;

async function createDetector() {
  switch (STATE.model) {
    case posedetection.SupportedModels.PoseNet:
      return posedetection.createDetector(STATE.model, {
        quantBytes: 4,
        architecture: 'MobileNetV1',
        outputStride: 16,
        inputResolution: {width: 500, height: 500},
        multiplier: 0.75
      });
    case posedetection.SupportedModels.BlazePose:
      const runtime = STATE.backend.split('-')[0];
      if (runtime === 'mediapipe') {
        console.log("mpPose.VERSION", mpPose.VERSION)
        return posedetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          solutionPath: '/dist/models/blaze'
          // solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${mpPose.VERSION}`
        });
      } else if (runtime === 'tfjs') {
        return posedetection.createDetector(
            STATE.model, {runtime, modelType: STATE.modelConfig.type});
      }
    case posedetection.SupportedModels.MoveNet:
      let modelType;
      let modelConfig = {}
      if (STATE.modelConfig.type == 'lightning') {
        modelType = posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING;
        modelConfig.modelUrl = "/dist/models/lightning/model.json"
      } else if (STATE.modelConfig.type == 'thunder') {
        modelType = posedetection.movenet.modelType.SINGLEPOSE_THUNDER;
        modelConfig.modelUrl = "/dist/models/thunder/model.json"
      } else if (STATE.modelConfig.type == 'multipose') {
        modelType = posedetection.movenet.modelType.MULTIPOSE_LIGHTNING;
        modelConfig.modelUrl = "/dist/models/multi-pose-lightning/model.json"
      }
      modelConfig.modelType = modelType;
      console.log('modelURL', modelConfig.modelUrl)
      // const modelConfig = {modelType};
      // modelConfig.modelUrl = 'http://localhost:9980/dist/models/lightning'
      // modelConfig.modelUrl = "/dist/models/lightning.json"
      if (STATE.modelConfig.customModel !== '') {
        modelConfig.modelUrl = STATE.modelConfig.customModel;
      }
      if (STATE.modelConfig.type === 'multipose') {
        modelConfig.enableTracking = STATE.modelConfig.enableTracking;
      }
      return posedetection.createDetector(STATE.model, modelConfig);
  }
}

async function checkGuiUpdate() {
  if (STATE.isTargetFPSChanged || STATE.isSizeOptionChanged) {
    camera = await Camera.setupCamera(STATE.camera);
    STATE.isTargetFPSChanged = false;
    STATE.isSizeOptionChanged = false;
  }

  if (STATE.ws.isConnectWebSocketChanged) {
    ws.close();
    setupWebSocket(STATE.ws.wsURL);
    STATE.ws.isConnectWebSocketChanged = false;
  }

  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    if (detector != null) {
      detector.dispose();
    }

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    try {
      detector = await createDetector(STATE.model);
    } catch (error) {
      detector = null;
      alert(error);
    }

    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

function beginEstimatePosesStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimatePosesStats() {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    stats.customFpsPanel.update(
        1000.0 / averageInferenceTime, 120 /* maxValue */);
    lastPanelUpdate = endInferenceTime;
  }
}

async function renderResult() {
  if (camera.video.readyState < 2) {
    await new Promise((resolve) => {
      camera.video.onloadeddata = () => {
        resolve(video);
      };
    });
  }

  let poses = null;

  // Detector can be null if initialization failed (for example when loading
  // from a URL that does not exist).
  if (detector != null) {
    // FPS only counts the time it takes to finish estimatePoses.
    beginEstimatePosesStats();

    // Detectors can throw errors, for example when using custom URLs that
    // contain a model that doesn't provide the expected output.
    try {
      poses = await detector.estimatePoses(
          camera.video,
          {maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false});
    } catch (error) {
      detector.dispose();
      detector = null;
      alert(error);
    }

    endEstimatePosesStats();
  }
  if(STATE.camera.displayCanvas) {
    camera.drawCtx();
  }
  

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old model,
  // which shouldn't be rendered.
  if (poses && poses.length > 0 && !STATE.isModelChanged) {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(poses)); 
    }
    if(STATE.camera.displayCanvas) {
      camera.drawResults(poses);
    }
  }
}

async function renderPrediction() {
  await checkGuiUpdate();

  if (!STATE.isModelChanged) {
    await renderResult();
  }

  rafId = requestAnimationFrame(renderPrediction);
};

async function app() {
  webcamDevices = await getWebcamDevices();
  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);
  if (!urlParams.has('model')) {
    urlParams.model = 'movenet'
    // alert('Cannot find model in the query string.');
    // return;
  }

  let wsURL = 'wss://new-socket-server.herokuapp.com/:443'
  if (urlParams.has('wsURL')) {
    wsURL = urlParams.get('wsURL')
  } 
  
  let webcamId = null;
  if (urlParams.has('webcamId')) {
    webcamId = urlParams.get('webcamId');
    if (checkDeviceIds(webcamId, webcamDevices)) {
      STATE.camera.deviceId.exact = webcamId;
    }
  } 
  setupWebSocket(wsURL);
  await setupDatGui(urlParams);
  
  console.log('webcamdevices1', webcamDevices)
  stats = setupStats();

  camera = await Camera.setupCamera(STATE.camera);

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);

  detector = await createDetector();

  console.log('starting');

  renderPrediction();

  
};

async function getWebcamDevices() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const webcams = devices.filter(device => device.kind === 'videoinput');
    console.log('webcams', webcams)
    return webcams.map(({ deviceId, label }) => ({ deviceId, label }));
  } catch (error) {
    console.error('Error getting webcam devices:', error);
    return [];
  }
}

function checkDeviceIds(key, deviceIds) {
  for (let i = 0; i < deviceIds.length; i++) {
    if (deviceIds[i].deviceId === key) {
      return true;
    }
  }
  return false;
}

function setupWebSocket(socketURL) {
  ws = new WebSocket(socketURL);

  ws.addEventListener('open', (event) => {
    console.log('WebSocket connection opened:', event);
    ws.send('pong');
    
    getWebcamDevices().then(devices => {
      console.log('devices', devices)
      ws.send(JSON.stringify({ type: 'webcamDevices', devices }));
    });
  });


  ws.addEventListener('message', async (event) => {
    if (event && event.data && event.data === "ping") {
      ws.send("pong");
      return;
    }
    const message = JSON.parse(event.data);
    if (message.type === 'selectWebcam') {
      console.log("GOT selectwebcam message", message)
      const deviceId = message.deviceId;
      // const videoElement = document.createElement('video');
      if (checkDeviceIds(deviceId, webcamDevices)) {
        STATE.camera.deviceId.exact = deviceId;
      }
      camera = await Camera.setupCamera(STATE.camera);
    }
    else if (message.type === 'selectModel') {
      
      // const modelType = message.modelType;
      
      // STATE.modelConfig.type = modelType;
      // STATE.isModelChanged = true;

      const modelType = message.modelType;
      const modelVersion = message.modelVersion;
      console.log("GOT selectModel message", message, modelType, modelVersion)
      // Map the received model type to the actual model.
      let mappedModel;
      switch (modelType) {
        case 'PoseNet':
          mappedModel = posedetection.SupportedModels.PoseNet;
          STATE.modelConfig = POSENET_CONFIG;
          break;
        case 'BlazePose':
          mappedModel = posedetection.SupportedModels.BlazePose;
          BLAZEPOSE_CONFIG.type = modelVersion;
          STATE.modelConfig = BLAZEPOSE_CONFIG;
          STATE.backend = 'mediapipe-gpu';
          STATE.isBackendChanged = true;
          break;
        case 'MoveNet':
          mappedModel = posedetection.SupportedModels.MoveNet;
          STATE.backend = 'tfjs-webgl';
          STATE.isBackendChanged = true;
          if (modelVersion === 'multipose') {
            // Set specific configurations for multi-pose.
            MOVENET_CONFIG.enableTracking = true;
            MOVENET_CONFIG.maxPoses = 6;  // You can adjust this value based on your requirement.
          } else {
            // Default configurations for single pose.
            MOVENET_CONFIG.enableTracking = false;
            MOVENET_CONFIG.maxPoses = 1;
          }
          
          MOVENET_CONFIG.type = modelVersion;
          STATE.modelConfig = MOVENET_CONFIG;
          break;
        default:
          console.error(`Unknown model type received: ${modelType}`);
          return;
      }
  
      // Update the STATE.
      STATE.model = mappedModel;
      STATE.isModelChanged = true;
    }
  });  
  
  // ws.onmessage = message => {
  //   if (message && message.data) {
  //     if (message.data === "ping") {
  //       console.log("got ping");
  //       ws.send("pong");
  //       return;
  //     }
  //   }
  // }

  ws.addEventListener('error', (event) => {
    console.error('Error in websocket connection', error);
  });

  ws.addEventListener('close', (event) => {
    console.log('Socket connection closed');
  });
  // ws.onclose = event => {
  //   console.log('Socket connection closed');
  //   alert('closing socket server');
  //   // clearInterval(keepAliveId);
  // }
}

app();
