scalaVersion := "2.13.1"

libraryDependencies ++= Seq(
  "org.tensorflow" % "tensorflow" % "1.13.1",
  "io.github.picnicml" %% "doddle-model" % "0.0.1-SNAPSHOT",
  "org.scalanlp" %% "breeze-natives" % "1.0",
  "org.slf4j" % "slf4j-nop" % "1.7.26",
)
