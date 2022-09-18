# Manual implementation of a bach gradient descent algorithm
In this repository, a batch gradient descent algorithm has been implemented in order to optimizate the parameters of a simple linear regression model.
## Algorithm general idea
The most general idea of the gradient descent algorithm (GD) is to tweak parameters iteratively in order to miniminize the cost function of a model.
Because we choose to optimice a linear regression model, the cost function used for measure the performance of the model is the mean square error (MSE).

$$
\text{MSE} = \frac{1}{m}\sum_{i=1}^m (\hat{y}_i - y_i)^2
$$

The main goal is to decrease the cost function until it reaches the minimum. To do that, the algorithm computes the gradient of the MSE. This give us the direction
in which the cost function increases, so we change the parameters in the contrary direction. 
$$\nabla_\theta \text{MSE}(\theta) = \frac{2}{m} \textbf{X} (\textbf{X\}\theta-\textbf{y})$$

$$\theta_{new} = \theta_{old} - \eta \nabla_\theta \text{MSE}(\theta),$$

where $\theta$ are the parameters of the model, $y$ are the target values, $\hat{y}$ are the predicted values and $\eta$ is the learning rate; a parameter of the algorithm
use for regulate how "fast" we change the parameters each iteration.

## Prediction of salary example
In the example proposed in the PredictSalary.py script, we show how to implement the algorithm in BGDLineaRegressor.py file.
The script plots the model predictions of salary as a function of the years of experience on top of the dataset points. At the same time, it
shows the root of the MSE (RMSE), so we can try to play with the model parameters (learning_rate and n_iterations) and watch how it changes the model and it adjustment
to the data (RMSE).


## References
Aurlien Gron. 2017. Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems (1st. ed.). O'Reilly Media, Inc.

