package io.picnicml.doddlemodel.spamham

import io.picnicml.doddlemodel.loadEstimator
import io.picnicml.doddlemodel.pipeline.Pipeline
import io.picnicml.doddlemodel.syntax.PredictorSyntax._

import scala.io.StdIn.readLine

object DeploySpamClassifier extends App {
  val sentenceEmbedder = new UniversalSentenceEncoder()
  val spamClassifier = loadEstimator[Pipeline]("logreg.model")
  while(true) {
    val sms = readLine("Input your SMS here: ")
    val yPred = spamClassifier.predict(sentenceEmbedder.embed(Array(sms)))
    if (yPred(0) == 1.0)
      println(s"SMS '$sms' marked as spam")
    else
      println(s"SMS '$sms' marked as a legitimate message")
  }
  sentenceEmbedder.close()
}
