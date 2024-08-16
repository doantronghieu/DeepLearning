from typing import Any, Callable, Dict, List, Optional, Type, Union
from mmengine.registry import (
    build_from_cfg, build_runner_from_cfg, build_model_from_cfg, build_scheduler_from_cfg,
    MODELS, RUNNERS, RUNNER_CONSTRUCTORS, LOOPS, HOOKS, STRATEGIES, DATASETS, DATA_SAMPLERS,
    TRANSFORMS, MODEL_WRAPPERS, WEIGHT_INITIALIZERS, OPTIMIZERS, OPTIM_WRAPPERS,
    OPTIM_WRAPPER_CONSTRUCTORS, PARAM_SCHEDULERS, METRICS, EVALUATOR, TASK_UTILS,
    VISUALIZERS, VISBACKENDS, LOG_PROCESSORS, INFERENCERS, FUNCTIONS,
    Registry, init_default_scope, DefaultScope
)
from mmengine.registry import count_registered_modules, traverse_registry_tree

class RegistryManager:
    def __init__(self):
        self.registry_dict = {
            'MODELS': MODELS,
            'RUNNERS': RUNNERS,
            'RUNNER_CONSTRUCTORS': RUNNER_CONSTRUCTORS,
            'LOOPS': LOOPS,
            'HOOKS': HOOKS,
            'STRATEGIES': STRATEGIES,
            'DATASETS': DATASETS,
            'DATA_SAMPLERS': DATA_SAMPLERS,
            'TRANSFORMS': TRANSFORMS,
            'MODEL_WRAPPERS': MODEL_WRAPPERS,
            'WEIGHT_INITIALIZERS': WEIGHT_INITIALIZERS,
            'OPTIMIZERS': OPTIMIZERS,
            'OPTIM_WRAPPERS': OPTIM_WRAPPERS,
            'OPTIM_WRAPPER_CONSTRUCTORS': OPTIM_WRAPPER_CONSTRUCTORS,
            'PARAM_SCHEDULERS': PARAM_SCHEDULERS,
            'METRICS': METRICS,
            'EVALUATOR': EVALUATOR,
            'TASK_UTILS': TASK_UTILS,
            'VISUALIZERS': VISUALIZERS,
            'VISBACKENDS': VISBACKENDS,
            'LOG_PROCESSORS': LOG_PROCESSORS,
            'INFERENCERS': INFERENCERS,
            'FUNCTIONS': FUNCTIONS
        }

    def get_registry(self, name: str) -> Registry:
        return self.registry_dict[name]

    def build(self, cfg: Dict[str, Any], registry_name: str, *args: Any, **kwargs: Any) -> Any:
        registry = self.get_registry(registry_name)
        return registry.build(cfg, *args, **kwargs)

    def build_from_cfg(self, cfg: Dict[str, Any], registry: Registry, default_args: Optional[Dict[str, Any]] = None) -> Any:
        return build_from_cfg(cfg, registry, default_args)

    def build_runner(self, cfg: Dict[str, Any]) -> Any:
        return build_runner_from_cfg(cfg, RUNNERS)

    def build_model(self, cfg: Union[Dict[str, Any], List[Dict[str, Any]]], default_args: Optional[Dict[str, Any]] = None) -> Any:
        return build_model_from_cfg(cfg, MODELS, default_args)

    def build_scheduler(self, cfg: Dict[str, Any], default_args: Optional[Dict[str, Any]] = None) -> Any:
        return build_scheduler_from_cfg(cfg, PARAM_SCHEDULERS, default_args)

    def register_module(self, registry_name: str, name: Optional[str] = None, 
                        force: bool = False, module: Optional[Union[Type, Callable]] = None) -> Callable:
        registry = self.get_registry(registry_name)
        return registry.register_module(name=name, force=force, module=module)

    def init_default_scope(self, scope: str) -> None:
        init_default_scope(scope)

    def get_current_scope(self) -> Optional[str]:
        return DefaultScope.get_current_instance().scope_name if DefaultScope.get_current_instance() else None

    def count_registered_modules(self, save_path: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
        return count_registered_modules(save_path, verbose)

    def traverse_registry_tree(self, registry_name: str, verbose: bool = False) -> List[Dict[str, Any]]:
        registry = self.get_registry(registry_name)
        return traverse_registry_tree(registry, verbose)

"""
# Usage example
registry_manager = RegistryManager()

# Initialize default scope
registry_manager.init_default_scope('myproject')

# Register a model
@MODELS
class MyModel:
    def __init__(self, param1: int, param2: str):
        self.param1 = param1
        self.param2 = param2

    def __repr__(self):
        return f"MyModel(param1={self.param1}, param2='{self.param2}')"

# Build a model
model_cfg = {"type": "MyModel", "param1": 42, "param2": "hello"}
model = registry_manager.build(model_cfg, 'MODELS')
print(model)  # Output: MyModel(param1=42, param2='hello')

# Register and build a runner
@RUNNERS
class MyRunner:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def run(self):
        print(f"Running with {self.model}")

runner_cfg = {
    "type": "MyRunner",
    "model": model_cfg,
    "optimizer": {"type": "SGD", "lr": 0.01}
}
runner = registry_manager.build_runner(runner_cfg)


# Count registered modules
stats = registry_manager.count_registered_modules(verbose=True)
print(stats)

# Traverse registry tree
tree = registry_manager.traverse_registry_tree('MODELS', verbose=True)
print(tree)

# Get current scope
current_scope = registry_manager.get_current_scope()
print(f"Current scope: {current_scope}")
"""