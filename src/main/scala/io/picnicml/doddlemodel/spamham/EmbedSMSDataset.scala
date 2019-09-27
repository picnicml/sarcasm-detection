package io.picnicml.doddlemodel.spamham

import cats.syntax.option._
import com.github.tototoshi.csv.{CSVReader, CSVWriter}
import io.picnicml.doddlemodel.data.Features

object EmbedSMSDataset extends App {
  val (sentences, y) = loadTextDataset()
  val embedder = new UniversalSentenceEncoder()
  val embeddings =  embedder.embed(sentences)
  embedder.close()
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

  def saveDataset(x: Features, y: Array[Double]): Unit = {
    val writer = CSVWriter.open(args(1))
    writer.writeRow("target" :: (0 until embedder.EMBEDDINGS_DIM).toList.map(i => s"e$i"))
    writer.writeRow("n" :: (0 until embedder.EMBEDDINGS_DIM).toList.map(_ => "n"))
    (0 until x.rows).foreach { rowIndex =>
      writer.writeRow(y(rowIndex) :: x(rowIndex, ::).t.toArray.toList)
    }
    writer.close()
  }
}
