export { authRouter } from "./auth";

export { 
  consciousnessRouter, 
  nearMissRouter, 
  attentionMetricsRouter, 
  ucpRouter,
  vocabularyRouter
} from "./consciousness";

export { 
  searchRouter, 
  formatRouter 
} from "./search";

export { oceanRouter } from "./ocean";

export { adminRouter } from "./admin";

export { default as olympusRouter } from "./olympus";

// Autonomic routes (full kernel + agency)
export { autonomicRouter, autonomicAgencyRouter } from "./autonomic";

// Immune system routes
export { immuneRouter } from "./immune";

// Training routes
export { trainingRouter } from "./training";

// Memory routes
export { memoryRouter } from "./memory";

// Feedback routes
export { feedbackRouter } from "./feedback";

// Coordizer routes
export { coordizerRouter } from "./coordizer";

export { federationRouter } from "./federation";

export { sscBridgeRouter } from "./ssc-bridge";

export { default as zettelkastenRouter } from "./zettelkasten";

export { default as pythonProxiesRouter } from "./python-proxies";
