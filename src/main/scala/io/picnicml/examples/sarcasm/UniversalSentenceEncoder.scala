package io.picnicml.examples.sarcasm

import breeze.linalg.DenseMatrix
import io.picnicml.doddlemodel.data.Features
import org.tensorflow.{SavedModelBundle, Tensor}

class UniversalSentenceEncoder {
  val EMBEDDINGS_DIM = 512
  private val model = SavedModelBundle.load("universal_sentence_encoder_large_v3", "serve")

  def embed(sentences: Array[String]): Features = {
    val computationResult = model
      .session()
      .runner()
      .feed("text_input", prepareInput(sentences))
      .fetch("embedded_text")
      .run()
      .get(0)
    getEmbeddings(computationResult, sentences.length)
  }

  private def prepareInput(sentences: Array[String]): Tensor[String] =
    Tensor.create(sentences.map(_.getBytes("UTF-8")), classOf[String])

  private def getEmbeddings(computationResult: Tensor[_], numExamples: Int): Features = {
    val memory = Array.ofDim[Array[Float]](numExamples)
    val embeddings = computationResult
      .copyTo(memory.map(_ => Array.ofDim[Float](EMBEDDINGS_DIM)))
      .map(_.map(_.toDouble))
    DenseMatrix(embeddings:_*)
  }

  def close(): Unit = model.session().close()
}
