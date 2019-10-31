package com.jerry.ann;

import com.jerry.util.ConsoleHelper;
import com.jerry.util.DataUtil;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.List;

/**
 * Hello world!
 *
 */
public class App
{
    public static void main(String[] args) throws Exception {
        if(args.length<5)
        {
            System.out.println("233,你的数据呢");
            return;
        }
        for (int i=0;i<args.length;i++)
        {
            System.out.println(args[i]);
        }
        ConsoleHelper helper=new ConsoleHelper(args);
        String trainfile=helper.getArg("-train","");
        String testfile=helper.getArg("-test","");
        String separator=helper.getArg("-sep",",");
        String outputfile=helper.getArg("-out","");
        float eta=helper.getArg("-eta",0.02f);
        int nIter=helper.getArg("-iter",5);

        DataUtil util=DataUtil.getInstance();
        List<DataNode> trainList=util.getDataList(trainfile,separator);
        List<DataNode> testList=util.getDataList(testfile,separator);
        //划分出数据集和训练集
        BufferedWriter output=new BufferedWriter(new FileWriter(new File(outputfile)));//划分输出集

        int typeCount=util.getmTypeCount();//花卉的种类
        AnnClassifier annClassifier=new AnnClassifier(trainList.get(0).getmAttribList().size(),trainList.get(0).getmAttribList().size()+8,typeCount);

        annClassifier.setTrainNodes(trainList);

        annClassifier.train(eta,nIter);
        for(int i=0;i<testList.size();i++)
        {
            DataNode test=testList.get(i);
            int type=annClassifier.test(test);
            List<Float> attribs=test.getmAttribList();
            for(int n=0;n<attribs.size();n++)
            {
                output.write(attribs.get(n)+",");
                output.flush();
            }
            output.write(util.getTypeName(type)+"\n");
            output.flush();
        }
        output.close();
    }

}
