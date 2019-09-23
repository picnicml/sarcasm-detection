package io.picnicml.doddlemodel.spamham

import breeze.linalg.DenseMatrix
import cats.syntax.option._
import com.github.tototoshi.csv.{CSVReader, CSVWriter}
import io.picnicml.doddlemodel.data.Features
import org.tensorflow.{SavedModelBundle, Tensor}

object EmbedDataset extends App {
  val EMBEDDINGS_DIM = 512

  val (sentences, y) = loadTextDataset()
  val model = SavedModelBundle.load("universal_sentence_encoder_large_v3", "serve")
  val embeddings = embed(sentences)
  model.session().close()

  println(s"shape of the embeddings matrix: (${embeddings.rows}, ${embeddings.cols})")
  saveDataset(embeddings, y)

  def loadTextDataset(): (Array[String], Array[Double]) = {
    val reader = CSVReader.open(args(0))
    // skip header line
    reader.readNext

    val accInit = (List[String](), List[Double]())
    val (text, y) = reader.toStream.foldRight(accInit) { case (rowValues, (text, y)) =>
      (rowValues(1) :: text,
       rowValues(0).some.map {
         case "spam" => 1.0
         case "ham" => 0.0
       }.getOrElse(Double.NaN) :: y)
    }

    reader.close()
    (text.toArray, y.toArray)
  }

  def embed(sentences: Array[String]): Features = {
    val computationResult = model
      .session()
      .runner()
      .feed("text_input", prepareInput(sentences))
      .fetch("embedded_text")
      .run()
      .get(0)
    getEmbeddings(computationResult)
  }

  private def prepareInput(sentences: Array[String]): Tensor[String] =
    Tensor.create(sentences.map(_.getBytes("UTF-8")), classOf[String])

  private def getEmbeddings(computationResult: Tensor[_]): Features = {
    val embeddings = computationResult
      .copyTo(sentences.map(_ => Array.fill[Float](EMBEDDINGS_DIM) { 0 }))
      .map(_.map(_.toDouble))
    DenseMatrix(embeddings:_*)
  }

  def saveDataset(x: Features, y: Array[Double]): Unit = {
    val writer = CSVWriter.open(args(1))
    writer.writeRow("target" :: (0 until EMBEDDINGS_DIM).toList.map(i => s"e$i"))
    writer.writeRow("n" :: (0 until EMBEDDINGS_DIM).toList.map(_ => "n"))

    (0 until x.rows).foreach { rowIndex =>
      writer.writeRow(y(rowIndex) :: x(rowIndex, ::).t.toArray.toList)
    }

    writer.close()
  }
}
