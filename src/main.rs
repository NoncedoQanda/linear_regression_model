use burn::tensor::{Tensor, Data, Int, Float};
use burn::backend::ndarray::NdArray;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{Sgd, SgdConfig};
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::train::{RegressionOutput, TrainOutput, TrainStep, LearnerBuilder};
use rand::distributions::{Distribution, Uniform};
use textplots::{Chart, Plot, Shape};

// Define the backend
type B = NdArray<f32>;

// Model definition
#[derive(Module, Debug)]
struct LinearRegression<B: Backend<Float = f32>> {
    linear: Linear<B>,
}

impl<B: Backend<Float = f32>> LinearRegression<B> {
    fn new() -> Self {
        let linear = LinearConfig::new(1, 1).init();
        Self { linear }
    }

    fn forward(&self, x: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
        self.linear.forward(x)
    }
}

// Training step implementation
impl<B: Backend<Float = f32>> TrainStep<RegressionOutput<B>, Tensor<B, 2, Float>> for LinearRegression<B> {
    fn step(&self, item: RegressionOutput<B>) -> TrainOutput<Tensor<B, 2, Float>> {
        let prediction = self.forward(item.inputs);
        let loss = ((prediction - item.targets).powf(2.0)).mean();
        TrainOutput::new(loss, prediction)
    }
}

fn main() {
    // Generate synthetic data
    let x = generate_x_data();
    let y = generate_y_data(&x);

    // Reshape for training (needs 2D tensors)
    let x_train = x.reshape([100, 1]);
    let y_train = y.reshape([100, 1]);

    // Initialize model and optimizer
    let model = LinearRegression::<B>::new();
    let optimizer = Sgd::new(&SgdConfig { lr: 0.01 });

    // Train the model
    let mut learner = LearnerBuilder::new()
        .model(model)
        .optimizer(optimizer)
        .build();

    for epoch in 1..=100 {
        let output = learner.step(RegressionOutput {
            inputs: x_train.clone(),
            targets: y_train.clone(),
        });
        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {}", epoch, output.loss);
        }
    }

    // Test the model
    let x_test = Tensor::<B, 1, Float>::arange(0.0..10.0, 0.1).reshape([100, 1]);
    let y_pred = learner.model.forward(x_test.clone());

    // Plot results
    let points: Vec<(f32, f32)> = x_test
        .into_data().value.into_iter()
        .zip(y_pred.into_data().value.into_iter())
        .map(|(x, y)| (x, y))
        .collect();
    Chart::new(120, 60, 0.0, 10.0)
        .lineplot(&Shape::Lines(&points))
        .display();
}

fn generate_x_data() -> Tensor<B, 1, Float> {
    let dist = Uniform::new(0.0, 10.0);
    let mut rng = rand::thread_rng();
    let values: Vec<f32> = dist.sample_iter(&mut rng).take(100).collect();
    Tensor::from_data(Data::new(values, [100].into()))
}

fn generate_y_data(x: &Tensor<B, 1, Float>) -> Tensor<B, 1, Float> {
    let mut rng = rand::thread_rng();
    let noise_dist = Uniform::new(-0.1, 0.1);
    let noise: Vec<f32> = noise_dist.sample_iter(&mut rng).take(100).collect();
    x.clone() * 2.0 + 1.0 + Tensor::from_data(Data::new(noise, [100].into()))
}