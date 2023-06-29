# Tape

A tiny (8 line) parameter tape implementation.


## Implementation

```py
class Tape:

  def parameter(self, x):
    if isinstance(self.tape, set):
      return self.tape.add(x) or x
    return next(self.tape)

  def run(self, f, x, tape):
    self.tape = tape and tape or iter(tape)
    return f(x), tape
```


## Usage

Define a `Linear` layer.

```py
def Linear(x, y):
  w = tape.parameter(np.random.randn(x, y))  # weight
  b = tape.parameter(np.random.randn(y))     # bias

  return lambda x: w.T @ x + b
```

Use the layer within a larger model.

```py
def model(x):
  x = relu(Linear(28 * 28, 128)(x))
  x = relu(Linear(128, 64)(x))
  x = softmax(Linear(64, 10)(x))

  return x
```

Run the model. Firstly initialize its parameters and then run it with those parameters.

```py
>>> batch = np.ones(28 * 28)
>>> x, parameters = tape.run(model, batch, set())       # Initialization. 
>>> x, parameters = tape.run(model, batch, parameters)  # Forward pass.
```

Compare with Flax.

```py
>>> batch = jnp.ones(28 * 28)
>>> parameters = model.init(jax.random.PRNGKey(0), batch)  # Initialization.
>>> x = model.apply(parameters, batch)                     # Foward pass.
```
