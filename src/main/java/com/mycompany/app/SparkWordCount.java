import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.SparkConf;
import org.apache.spark.util.StatCounter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Collection;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.lang.Math.*;

public final class SparkWordCount {

    public static void main(String[] args) throws Exception {

        // create Spark context with Spark configuration
        JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName("wordcount"));

        // set the input files
        JavaPairRDD<String, String> FilenameContentRDD = sc.wholeTextFiles("datafiles/");
        Long FileCounts = Files.list(Paths.get("datafiles/")).count();
        System.err.println("\n\n******* file count:\n"+FileCounts);
        List <String> StopWords = Arrays.asList(String.join("\n", Files.readAllLines(Paths.get("stopwords.txt"))).split("\n"));
        System.err.println("\n\n******* stopword count:\n"+StopWords.size());
        List <String> QueryWords = Arrays.asList(String.join("\n", Files.readAllLines(Paths.get("query.txt"))).split("\\W+"));
        System.err.println("\n\n******* queryword count:\n"+QueryWords.size());

        // Step 1 #################################
        // Flapmap through all files' content (values) using flatmapvalues
        JavaPairRDD<String, Integer> count_all = FilenameContentRDD
                .flatMapValues(content -> new ArrayList <String> (Arrays.asList(content.toLowerCase().split("\\W+"))))
                // Filter out stopwords
                .filter(tuple -> !StopWords.contains(tuple._2()))
                // Generate new key value pairs (word@@file,1)
                .mapToPair(tuple -> new Tuple2<>(tuple._2()+"@@"+tuple._1.split("/")[tuple._1.split("/").length-1], 1))
                // Word count
                .reduceByKey((a, b) -> a + b);

//        // Review Step 1
//        count_all.foreach(data -> {
//            System.err.println("******model="+data._1() + " label=" + data._2());
//        });

        // Step 2 #################################
        JavaPairRDD<String, String> temp_all = count_all
                .mapToPair(tuple -> new Tuple2<>(tuple._1().split("@@")[0],tuple._1().split("@@")[1]+"=="+tuple._2()));

        JavaPairRDD<String, Integer> doc_count = temp_all.mapToPair(tuple -> new Tuple2<>(tuple._1(), 1)).reduceByKey((a,b) -> a+b);
        JavaPairRDD<String, Tuple2<String, Integer>> agg_all = temp_all.join(doc_count);

        // Calculate TF-IDF values = (fre + ln(freq))*ln(TotalFileCounts/doccounts)
        JavaPairRDD<String, Double> TFIDF_all = agg_all
                .mapToPair( new PairFunction <Tuple2<String, Tuple2<String, Integer>>, String, Double> () {
                    @Override
                    public Tuple2<String, Double> call(Tuple2<String, Tuple2<String, Integer>> tuple) {
                        double freq = Double.parseDouble((tuple._2())._1().split("==")[1]);
                        double doccounts = (double) ((tuple._2())._2());
                        double TFIDF = (freq + java.lang.Math.log1p(freq)) / java.lang.Math.log1p(FileCounts / doccounts);
                        return (new Tuple2(tuple._1()+"@@"+(tuple._2())._1().split("==")[0], TFIDF));
                    }
                });

//        // Review Step 2
//        TFIDF_all.foreach(data -> {
//            System.err.println("******model2="+data._1() + " label2=" + (data._2()));
//        });

        // Step 3 #################################
        JavaPairRDD<String, String> doc_TFIDF_map = TFIDF_all
                .mapToPair(tuple -> new Tuple2<>(tuple._1().split("@@")[1], tuple._1().split("@@")[0]+"=="+tuple._2()));

        JavaPairRDD<String, Double> doc_TFIDF_var = doc_TFIDF_map
                .mapToPair(tuple -> new Tuple2<>(tuple._1(), Double.parseDouble(tuple._2().split("==")[1])))
                .reduceByKey((a, b) -> a + b * b)
                .mapToPair(tuple -> new Tuple2<>(tuple._1(), java.lang.Math.sqrt(tuple._2())));

        JavaPairRDD<String, Tuple2<String, Double>> norm_agg_all = doc_TFIDF_map.join(doc_TFIDF_var);

        // Calculate normalized TF-IDF values = (TFIDF value / TFIDF variance)
        JavaPairRDD<String, Double> TFIDF_norm = norm_agg_all
                .mapToPair( new PairFunction <Tuple2<String, Tuple2<String, Double>>, String, Double> () {
                    @Override
                    public Tuple2<String, Double> call(Tuple2<String, Tuple2<String, Double>> tuple) {
                        double variance = (double) ((tuple._2())._2());
                        double TFIDF = Double.parseDouble((tuple._2())._1().split("==")[1]);
                        double TFIDF_norm = (double) (TFIDF / variance);
                        String keyword_doc = (tuple._2())._1().split("==")[0] + "@@" + tuple._1();
                        return (new Tuple2(keyword_doc, TFIDF_norm));
                    }
                });

//        // Review Step 3
//        TFIDF_norm.foreach(data -> {
//            System.err.println("******model3="+data._1() + " label3=" + (data._2()));
//        });

        // Step 4 #################################
        // Filter query words, map to document as key, calculate sum of TFIDF per document
        JavaPairRDD<String, Double> doc_TFIDF_sum = TFIDF_norm
                .filter(tuple -> QueryWords.contains(tuple._1().split("@@")[0]))
                .mapToPair(tuple -> new Tuple2<>(tuple._1().split("@@")[1], tuple._1().split("@@")[0] + "==" + tuple._2()))
                .aggregateByKey((double) 0, (a, b) -> {
                    return (double) a + Double.parseDouble(b.split("==")[1]);
                }, (a, b) -> a + b);

//        // Review Step 4
//        doc_TFIDF_sum.foreach(data -> {
//            System.err.println("******model4="+data._1() + " label4=" + (data._2()));
//        });

        // Step 5 #################################
        JavaPairRDD<Double, String> sorted_docs = doc_TFIDF_sum
                .mapToPair(tuple -> new Tuple2<>(tuple._2(), tuple._1()))
                .sortByKey(false);

        // System.err.println("\n\n\n\n\n*******Result: "+sorted_docs.take(3));
        // result.saveAsTextFile("outfile2");
        JavaPairRDD<String, Double> result = JavaPairRDD.fromJavaRDD(sc.parallelize(sorted_docs.take(3)))
                .mapToPair(tuple -> new Tuple2<>(tuple._2(), tuple._1()));
        result.saveAsTextFile("outfile");
    }
}
