package io.picnicml.examples.sarcasm

import breeze.linalg.sum
import io.picnicml.doddlemodel.data.DatasetUtils.{shuffleDataset, splitDataset}
import io.picnicml.doddlemodel.data.Feature.FeatureIndex
import io.picnicml.doddlemodel.data.{Features, Target, TrainTestSplit}
import io.picnicml.doddlemodel.linear.LogisticRegression
import io.picnicml.doddlemodel.metrics.f1Score
import io.picnicml.doddlemodel.modelselection.{CrossValidation, HyperparameterSearch, KFoldSplitter}
import io.picnicml.doddlemodel.pipeline.Pipeline
import io.picnicml.doddlemodel.pipeline.Pipeline.pipe
import io.picnicml.doddlemodel.preprocessing.StandardScaler
import io.picnicml.doddlemodel.syntax.PredictorSyntax._
import io.picnicml.examples.sarcasm.EmbeddedDataSerialization.{loadSerializedDataset, printDatasetInfo}

object TrainClassifier extends App {

  val (x, y, featureIndex) = loadSerializedDataset(args(0), args(1), args(2))
  printDatasetInfo(x, y, featureIndex)
  val split = shuffleSplitData(x, y)
  val selectedModel = gridSearch(split, featureIndex)
  val score = f1Score(split.yTe, selectedModel.predict(split.xTe))
  println(f"Test F1 score of the selected model: $score%1.4f")
  selectedModel.save(args(3))

  def shuffleSplitData(x: Features, y: Target): TrainTestSplit = {
    println("Shuffling and splitting data")
    val (xShuffled, yShuffled) = shuffleDataset(x, y)
    val split = splitDataset(xShuffled, yShuffled, proportionTrain = 0.9f)
    println(s"Training set size: ${split.xTr.rows}, test set size: ${split.xTe.rows}")
    println(s"Proportion of sarcastic examples in training set: ${sum(split.yTr) / split.xTr.rows}")
    println(s"Proportion of sarcastic examples in test set: ${sum(split.yTe) / split.xTe.rows}\n")
    split
  }

  def gridSearch(split: TrainTestSplit, featureIndex: FeatureIndex): Pipeline = {
    val numGridSearchIterations = 30
    val cv = CrossValidation(f1Score, KFoldSplitter(numFolds = 10))
    val search = HyperparameterSearch(numGridSearchIterations, cv)
    val (start, end, step) = (1e-5, 5.0, (5.0 - 1e-5) / numGridSearchIterations)
    val grid = Range.BigDecimal(start, end, step).map(_.toFloat).iterator
    println("Searching the hyperparameter space")
    search.bestOf(split.xTr, split.yTr) { generateModel(lambda = grid.next) }
  }

  // lambda is L2 regularization strength
  def generateModel(lambda: Float): Pipeline =
    Pipeline(List(pipe(StandardScaler(featureIndex))))(pipe(LogisticRegression(lambda)))
}
