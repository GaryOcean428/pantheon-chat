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
  formatRouter,
  budgetRouter
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

export { default as zettelkastenRouter } from "./zettelkasten";

export { default as pythonProxiesRouter } from "./python-proxies";

// SSC Bridge - connects to SearchSpaceCollapse for Bitcoin recovery
export { sscBridgeRouter } from "./ssc-bridge";

// Pantheon Registry routes - formal god contracts and kernel spawner
export { default as pantheonRegistryRouter } from "./pantheon-registry";
