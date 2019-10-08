package io.picnicml.examples.sarcasm

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}

import breeze.linalg.sum
import io.picnicml.doddlemodel.data.CsvLoader.loadCsvDataset
import io.picnicml.doddlemodel.data.Feature.FeatureIndex
import io.picnicml.doddlemodel.data.{DatasetWithIndex, Features, Target}

object EmbeddedDataSerialization extends App {
  val (x, y, featureIndex) = loadCsvData()
  printDatasetInfo(x, y, featureIndex)

  serializeDataset(x, y, featureIndex)
  val (xLoaded, yLoaded, featureIndexLoaded) = loadSerializedDataset()
  printDatasetInfo(xLoaded, yLoaded, featureIndexLoaded)

  def loadCsvData(): DatasetWithIndex = {
    println("Loading csv data...")
    val (dataset, featureIndexOriginal) = loadCsvDataset(args(0))
    // 'target' is the first column, drop it from features and feature index
    val (x, y) = (dataset(::, 1 to -1), dataset(::, 0))
    val featureIndex = featureIndexOriginal.drop(0)
    (x, y, featureIndex)
  }

  def serializeDataset(x: Features, y: Target, featureIndex: FeatureIndex): Unit = {
    println("Saving serialized data...")
    serialize(x, args(1))
    serialize(y, args(2))
    serialize(featureIndex, args(3))
  }

  def serialize(obj: Any, filePath: String): Unit = {
    val outputStream = new ObjectOutputStream(new FileOutputStream(filePath))
    outputStream.writeObject(obj)
    outputStream.close()
  }

  def loadSerializedDataset(): DatasetWithIndex = {
    println("Loading serialized data...")
    val x = loadSerialized[Features](args(1))
    val y = loadSerialized[Target](args(2))
    val featureIndex = loadSerialized[FeatureIndex](args(3))
    (x, y, featureIndex)
  }

  def loadSerialized[A](filePath: String): A = {
    val inputStream = new ObjectInputStream(new FileInputStream(filePath))
    val obj = inputStream.readObject.asInstanceOf[A]
    inputStream.close()
    obj
  }

  def printDatasetInfo(x: Features, y: Target, featureIndex: FeatureIndex): Unit = {
    println(s"Features: $featureIndex")
    println(s"Shape of the feature matrix: (${x.rows}, ${x.cols})")
    println(s"Length of the target vector: (${y.length},)")
    println(s"Proportion of sarcastic examples: ${sum(y) / x.rows}\n")
  }
}
