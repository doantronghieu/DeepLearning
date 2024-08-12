# How to use container

Certainly! I'd be happy to explain how to use the DIContainer (Dependency Injection Container) in the enhanced version of the code. The DIContainer is a powerful tool for managing dependencies and creating instances of various components in your application. Here's a step-by-step guide on how to use it:

1. Create an instance of the DIContainer:

```python
container = DIContainer()
```

2. Register factories for your components:

```python
container.register('dataset_builder', lambda: DefaultDatasetBuilder())
container.register('model_builder', lambda: DefaultModelBuilder())
container.register('optimizer_builder', lambda: DefaultOptimizerBuilder())
container.register('strategy_builder', lambda: DefaultStrategyBuilder())
```

3. You can also register more complex factories that have dependencies:

```python
container.register('training_strategy', lambda: DefaultTrainingStrategy(DefaultOptimizationStrategy()))
```

4. Use the container in your BaseMMEngine subclass:

```python
class MyMMEngine(BaseMMEngine):
    def __init__(self, config: MMEngineConfig, container: DIContainer, event_manager: EventManager):
        super().__init__(config, container, event_manager)

    # Implement abstract methods...
```

5. Resolve dependencies when needed:

```python
class MyMMEngine(BaseMMEngine):
    # ...

    def build_dataset(self, dataset_config: BaseDatasetConfig) -> BaseDataset:
        dataset_builder = self.container.resolve('dataset_builder')
        return dataset_builder.build(dataset_config)

    def build_model(self) -> torch.nn.Module:
        model_builder = self.container.resolve('model_builder')
        return model_builder.build(self.config.model)

    # ...
```

6. Set up and run your experiment:

```python
config = MMEngineConfig(...)  # Your configuration
container = DIContainer()
event_manager = EventManager()

# Register your components
container.register('dataset_builder', lambda: DefaultDatasetBuilder())
container.register('model_builder', lambda: DefaultModelBuilder())
container.register('optimizer_builder', lambda: DefaultOptimizerBuilder())
container.register('strategy_builder', lambda: DefaultStrategyBuilder())

# Create your engine
engine = MyMMEngine(config, container, event_manager)

# Run the experiment
engine.run_experiment()
```

Here are some additional tips for using the DIContainer effectively:

1. Lazy Initialization: The lambda functions used in registration allow for lazy initialization of components. They are only created when resolved, which can be more efficient.

2. Singleton vs New Instance: If you want a component to be a singleton (same instance every time), you can create it once and return that instance in the factory:

```python
dataset_builder = DefaultDatasetBuilder()
container.register('dataset_builder', lambda: dataset_builder)
```

3. Parameterized Factories: You can create factories that accept parameters:

```python
container.register('optimized_model', lambda learning_rate: OptimizedModel(learning_rate))
# Usage:
optimized_model = container.resolve('optimized_model')(0.001)
```

4. Dependency Chains: You can resolve dependencies of dependencies:

```python
container.register('training_pipeline', lambda: TrainingPipeline(
    container.resolve('model_builder'),
    container.resolve('dataset_builder'),
    container.resolve('optimizer_builder')
))
```

5. Testing: The DIContainer makes it easy to swap out real implementations with mocks or stubs for testing:

```python
container.register('dataset_builder', lambda: MockDatasetBuilder())
```

By using the DIContainer, you make your code more modular and easier to test. It also provides a central place to manage the creation and lifecycle of your components, which can be particularly useful in large, complex applications.