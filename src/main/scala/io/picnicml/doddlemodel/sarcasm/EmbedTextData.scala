package io.picnicml.doddlemodel.sarcasm

import com.github.tototoshi.csv.{CSVReader, CSVWriter}

object EmbedTextData extends App {
  val (sentences, y) = loadTextDataset()
  val embedder = new UniversalSentenceEncoder()
  val writer = CSVWriter.open(args(1))
  writer.writeRow("target" :: (0 until embedder.EMBEDDINGS_DIM).toList.map(i => s"e$i"))
  writer.writeRow("n" :: (0 until embedder.EMBEDDINGS_DIM).toList.map(_ => "n"))

  println("Embedding started...")
  var persisted = 0
  sentences.zip(y).grouped(4096).foreach { batch =>
    val batchEmbedding = embedder.embed(batch.map(_._1))
    val batchY = batch.map(_._2)
    batch.indices.foreach { rowIndex =>
      writer.writeRow(batchY(rowIndex) :: batchEmbedding(rowIndex, ::).t.toArray.toList)
    }
    persisted += batch.length
    println(f"Progress: ${(persisted / sentences.length.toDouble) * 100}%2.2f%%")
  }
  println("Embedding completed")

  embedder.close()
  writer.close()

  def loadTextDataset(): (Array[String], Array[Double]) = {
    val reader = CSVReader.open(args(0))
    // skip header line
    reader.readNext

    val accInit = (List[String](), List[Double]())
    val (text, y) = reader.toStream.foldRight(accInit) { case (rowValues, (text, y)) =>
      (rowValues(1) :: text, rowValues(0).toDouble :: y)
    }

    reader.close()
    (text.toArray, y.toArray)
  }
}
