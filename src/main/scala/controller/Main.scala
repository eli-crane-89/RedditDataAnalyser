package controller

import org.mongodb.scala._
import org.mongodb.scala.model.Projections._
import org.mongodb.scala.model.Filters._
import Helpers._

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.AFTSurvivalRegression

import breeze.linalg._

import org.apache.spark.sql.SparkSession


object main extends App {
//  val ages = Seq(42, 75, 29, 85)
//  println(s"The oldest person is ${ages.max}")


  //Session
  val spark = SparkSession
    .builder()
    .config("spark.master", "local")
    //.config("spark.network.timeout", "10000s")
    //.config("spark.executor.heartbeatInterval", "5000s")
   // .config("spark.executor.heartbeat.maxFailures", "10000")
    //.config("spark.sql.broadcastTimeout", "36000")
    .getOrCreate()

  import spark.implicits._

  //Static Values
  val lowerScoreLimit = 100

  // To directly connect to the default server localhost on port 27017
  val mongoClient: MongoClient = MongoClient("mongodb://localhost:27017/")

  val database: MongoDatabase = mongoClient.getDatabase("Posts2")

  val collections: List[String] = List("news")
  //,"worldnews""philosophy","futurology","twoxchromosomes","science"

  val seqRelevantArticles: List[(String,Map[String, Seq[((Int, Double, Int, Double),Int)]])] = collections.map(colName => getRelevantArticlesByCollection(colName))

  val seqExpandedArticlesByRow = mapRelevantArticlesToFinalSequence(seqRelevantArticles)

  val seqRegression = mapArticlesToSeqForRegression(seqExpandedArticlesByRow)

  val dfRegression = seqRegression.map(seq=>(seq._3+1,seq._4,Vectors.dense(seq._5._1,seq._5._2,seq._5._2))).toDF("label","censor","features")

  dfRegression.take(5).foreach(ea=>println(ea))
  //seqExpandedArticlesByRow.takeRight(5).foreach(ea=>println(ea))


  val quantileProbabilities = Array(0.3, 0.6)
  val aft = new AFTSurvivalRegression()
    .setQuantileProbabilities(quantileProbabilities)
    .setQuantilesCol("quantiles")

  val model = aft.fit(dfRegression)
//
  // Print the coefficients, intercept and scale parameter for AFT survival regression
  println(s"There were a total of ${seqRegression.filter(_._4==1).length} censured articles.")
  println(s"Coefficients: ${model.coefficients}")
  println(s"Intercept: ${model.intercept}")
  println(s"Scale: ${model.scale}")
  model.transform(dfRegression).show(false)



  //stop spark
  spark.stop()

    def mapRelevantArticlesToFinalSequence(listRelevantArticlesByCollection : List[(String, Map[String,Seq[((Int, Double, Int, Double),Int)]])]): Seq[(String,String,Int,Double,Int,Double,Int)] = {

      for ((coll, mapArt) <- listRelevantArticlesByCollection; (id, values) <- mapArt; ((score,upvote_ratio,comms_num,time_inserted),step) <- values)
                              yield (coll, id, score, upvote_ratio, comms_num,time_inserted, step)

    }

    //: Seq[(String,String,Int,(Int,Double,Int))] =
    def mapArticlesToSeqForRegression(seqRelevantArticlesByCollection : Seq[(String,String,Int,Double,Int,Double,Int)]): Seq[(String,String,Int,Int,(Double,Double,Double))] =  {

      val groupByStep = seqRelevantArticlesByCollection.groupBy(_._7)

      groupByStep.take(5).foreach(step=>println(s"(${step._1},${step._2.head._3})"))


      val avg_score_by_step = groupByStep.map(step=>(step._1,step._2.map(_._3).foldRight(0.0)(_+_)/step._2.length))
      val avg_ratio_by_step = groupByStep.map(step=>(step._1,step._2.map(_._4).foldRight(0.0)(_+_)/step._2.length))
      val avg_comms_by_step = groupByStep.map(step=>(step._1,step._2.map(_._5).foldRight(0.0)(_+_)/step._2.length))

      //avg_score_by_step.foreach(step=>println(s"The Average Score for Step ${step._1} is ${step._2}"))
      //avg_ratio_by_step.foreach(step=>println(s"The Average Upvote Ratio for Step ${step._1} is ${step._2}"))
      //avg_comms_by_step.foreach(step=>println(s"The Average Comments for Step ${step._1} is ${step._2}"))

      val seqArticlesForRegression = seqRelevantArticlesByCollection.map(art=>(art._1, art._2, art._7, art._3 compare lowerScoreLimit match {case 0 => 1 case 1 => 1 case -1 => 0},
        (art._3 - avg_score_by_step.get(art._7).get, art._4 - avg_ratio_by_step.get(art._7).get, art._5 - avg_comms_by_step.get(art._7).get)))

      //Get Only First Censure Step
      val mapMinCensure = seqArticlesForRegression.filter(_._4 == 1).groupBy(_._2).map(gb=>(gb._1,gb._2.map(_._3).min))
      seqArticlesForRegression.filter(art=>(!mapMinCensure.exists(_._1 == art._2)) || mapMinCensure.get(art._2).get >= art._3)

    }


    def getRelevantArticlesByCollection(collectionName: String): (String, Map[String,Seq[((Int,Double,Int,Double),Int)]]) = {
      var collection: MongoCollection[Document] = database.getCollection(collectionName)
      println(s"The current collection is: ${collectionName}")
      println(s"Number of documents:  ${collection.countDocuments().results().head}")
      println(s"The top document is:")
      println(s"${collection.find().limit(1).printResults()}")

      val collSpecArt = collection.find(equal("id","b2lvmd")).projection(fields(include("id", "score", "upvote_ratio", "comms_num", "time_inserted"), excludeId()))
      val seqArticle = collSpecArt.map(doc => ((doc("id").asString.getValue), doc("score").asInt32.getValue, doc("upvote_ratio").asDouble.getValue, doc("comms_num").asInt32.getValue, doc("time_inserted").asDouble.getValue)).collect().results().head.sortBy(_._5)

      seqArticle.take(5).foreach(println)


      //Get Article collection and convert to list
      //equal("id","awa0en")
      val collArticles = collection.find().projection(fields(include("id", "score", "upvote_ratio", "comms_num", "time_inserted"), excludeId()))
      val seqArticles = collArticles.map(doc => ((doc("id").asString.getValue), doc("score").asInt32.getValue, doc("upvote_ratio").asDouble.getValue, doc("comms_num").asInt32.getValue, doc("time_inserted").asDouble.getValue)).collect().results().head.sortBy(_._5)
      val groupedArticles = seqArticles.groupBy(_._1).map(seqArt=>(seqArt._1,seqArt._2.map(art=>(art._2,art._3,art._4,art._5)).zipWithIndex))


      //Print for progress
      println(s"There were a total of ${seqArticles.length} recovered and a total of ${groupedArticles.size} unique articles retrieved.")
      println(s"The average length of time between articles ticks is: ${groupedArticles.map(ga=>determineAverageTickTime(ga)).foldRight(0.0)(_+_)/groupedArticles.size}")
      groupedArticles.take(5).foreach(ga=>println(s"(${ga._1},${ga._2})"))

      //Return
      (collectionName, groupedArticles.filter(ga=>determineArticleRelevancy(ga)))

    }

  def determineAverageTickTime(groupedArticle: (String, Seq[((Int,Double,Int,Double),Int)])) : Double = {

    val minTime = groupedArticle._2.head._1._4
    val mapTimesScores = groupedArticle._2.map(seq => ((seq._1._4 - minTime) / 60))

    (mapTimesScores zip mapTimesScores.drop(1)).map({case(ml,mr)=>mr-ml}).foldRight(0.0)(_ + _) / mapTimesScores.size

  }

    def determineArticleRelevancy(groupedArticle: (String, Seq[((Int,Double,Int,Double),Int)])): Boolean = {

      //Eliminates articles that were caught too late in process
      if (groupedArticle._2.head._1._3 >= 15) {
        return false
      }

      //Eliminates articles that do not have minimum number of steps
      if (groupedArticle._2.map(seq=>seq._2).seq.max < 30){
        return false
      }

      return true
    }

}
