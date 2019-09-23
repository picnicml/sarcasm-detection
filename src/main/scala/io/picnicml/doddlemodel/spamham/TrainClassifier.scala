package io.picnicml.doddlemodel.spamham

import java.io.File

import io.picnicml.doddlemodel.data.CsvLoader.loadCsvDataset

object TrainClassifier extends App {
  val (dataset, featureIndexOriginal) = loadCsvDataset(new File(args(0)))
  // 'target' is the first column, drop it from features and feature index
  val (x, y) = (dataset(::, 1 to -1), dataset(::, 0))
  val featureIndex = featureIndexOriginal.drop(0)

  println(featureIndex)
  println(s"shape of the feature matrix: (${x.rows}, ${x.cols})")
  println(s"length of the target vector: (${y.length},)")
}
