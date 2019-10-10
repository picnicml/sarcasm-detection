package io.picnicml.examples.sarcasm

import io.picnicml.doddlemodel.loadEstimator
import io.picnicml.doddlemodel.pipeline.Pipeline
import io.picnicml.doddlemodel.syntax.PredictorSyntax._

import scala.io.StdIn.readLine

object ClassifyDemo extends App {
  val embedder = new UniversalSentenceEncoder()
  val classifier = loadEstimator[Pipeline](args(0))
  while(true) {
    val input = readLine("Input your text:  ")
    val yPred = classifier.predict(embedder.embed(Array(input)))
    if (yPred(0) == 1.0)
      println(s"Text '$input' marked as SARCASTIC")
    else
      println(s"Text '$input' marked as REGULAR")
  }
}
