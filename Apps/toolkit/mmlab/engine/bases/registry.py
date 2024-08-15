from typing import Any, Callable, Dict, List, Optional, Type, Union
from mmengine import Registry, build_from_cfg
from mmengine.registry import build_runner_from_cfg, build_model_from_cfg, build_scheduler_from_cfg, MODELS, RUNNERS, RUNNER_CONSTRUCTORS, LOOPS, HOOKS, STRATEGIES, DATASETS, DATA_SAMPLERS, TRANSFORMS, MODEL_WRAPPERS, WEIGHT_INITIALIZERS, OPTIMIZERS, OPTIM_WRAPPERS, OPTIM_WRAPPER_CONSTRUCTORS, PARAM_SCHEDULERS, METRICS, EVALUATOR, TASK_UTILS, VISUALIZERS, VISBACKENDS, LOG_PROCESSORS, INFERENCERS, FUNCTIONS

from mmengine.registry import init_default_scope, DefaultScope
from mmengine.registry import count_registered_modules, traverse_registry_tree

class RegistryManager:
    def __init__(self):
        self.registries: Dict[str, Registry] = {}
        self.default_scope: Optional[str] = None

    def create_registry(
      self, name: str, 
      build_func: Optional[Callable] = None,
      parent: Optional[str] = None, 
      scope: Optional[str] = None, 
      locations: Optional[List[str]] = None
    ) -> Registry:
        parent_registry = self.registries[parent] if parent else None
        registry = Registry(name, build_func=build_func, parent=parent_registry, 
                            scope=scope, locations=locations)
        self.registries[name] = registry
        return registry

    def get_registry(self, name: str) -> Registry:
        return self.registries[name]

    def build(
      self, cfg: Dict[str, Any], registry_name: str, 
      *args: Any, **kwargs: Any
    ) -> Any:
        registry = self.get_registry(registry_name)
        return registry.build(cfg, *args, **kwargs)

    def build_from_cfg(
      self, cfg: Dict[str, Any], registry: Registry, 
      default_args: Optional[Dict[str, Any]] = None
    ) -> Any:
        return build_from_cfg(cfg, registry, default_args)

    def build_runner(
      self, cfg: Dict[str, Any], registry: Registry
    ) -> Any:
        return build_runner_from_cfg(cfg, registry)

    def build_model(
      self, cfg: Union[Dict[str, Any], List[Dict[str, Any]]], 
      registry: Registry, default_args: Optional[Dict[str, Any]] = None
    ) -> Any:
        return build_model_from_cfg(cfg, registry, default_args)

    def build_scheduler(
      self, cfg: Dict[str, Any], registry: Registry, 
      default_args: Optional[Dict[str, Any]] = None
    ) -> Any:
        return build_scheduler_from_cfg(cfg, registry, default_args)

    def register_module(
      self, registry_name: str, name: Optional[str] = None, 
      force: bool = False, module: Optional[Union[Type, Callable]] = None
    ) -> Callable:
        registry = self.get_registry(registry_name)
        return registry.register_module(name=name, force=force, module=module)

    def init_default_scope(self, scope: str) -> None:
        init_default_scope(scope)
        self.default_scope = scope

    def get_current_scope(self) -> Optional[str]:
        return DefaultScope.get_current_instance().scope_name if DefaultScope.get_current_instance() else None

    def switch_scope_and_registry(self, scope: str) -> Registry:
        for registry in self.registries.values():
            if registry.scope == scope:
                return registry
        raise KeyError(f"Cannot find scope {scope}")

    def count_registered_modules(self, save_path: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
        return count_registered_modules(save_path, verbose)

    def traverse_registry_tree(
      self, registry_name: str, verbose: bool = False
    ) -> List[Dict[str, Any]]:
        registry = self.get_registry(registry_name)
        return traverse_registry_tree(registry, verbose)

    def import_modules_from_locations(self, registry_name: str) -> None:
        registry = self.get_registry(registry_name)
        registry.import_from_location()

# Usage example
registry_manager = RegistryManager()

# Create registries
models_registry = registry_manager.create_registry(
  'models', build_func=build_from_cfg,
  scope='myproject', locations=['myproject.models']
)
runners_registry = registry_manager.create_registry(
  'runners', build_func=build_runner_from_cfg, 
  scope='myproject'
)
schedulers_registry = registry_manager.create_registry(
  'schedulers', build_func=build_scheduler_from_cfg,
  scope='myproject'
)

# Initialize default scope
registry_manager.init_default_scope('myproject')

# Register a model
@registry_manager.register_module('models')
class MyModel:
    def __init__(self, param1: int, param2: str):
        self.param1 = param1
        self.param2 = param2

    def __repr__(self):
        return f"MyModel(param1={self.param1}, param2='{self.param2}')"

# Build a model
model_cfg = {"type": "MyModel", "param1": 42, "param2": "hello"}
model = registry_manager.build(model_cfg, 'models')
print(model)  # Output: MyModel(param1=42, param2='hello')

# Register and build a runner
@registry_manager.register_module('runners')
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
runner = registry_manager.build_runner(runner_cfg, runners_registry)
runner.run()  # Output: Running with MyModel(param1=42, param2='hello')

# Use MODELS from mmengine
print(MODELS)  # This will print the MODELS registry from mmengine

# Count registered modules
stats = registry_manager.count_registered_modules(verbose=True)
print(stats)

# Traverse registry tree
tree = registry_manager.traverse_registry_tree('models', verbose=True)
print(tree)

# Import modules from locations
registry_manager.import_modules_from_locations('models')

# Switch scope
current_scope = registry_manager.get_current_scope()
print(f"Current scope: {current_scope}")

try:
    new_registry = registry_manager.switch_scope_and_registry('newscope')
except KeyError as e:
    print(f"Error: {e}")