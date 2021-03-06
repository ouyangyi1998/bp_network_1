package com.jerry.util;

import com.jerry.ann.DataNode;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.*;

public class DataUtil {
    private static DataUtil instance = null;
    private Map<String, Integer> mTypes;
    private int mTypeCount;

    private DataUtil() {
        mTypes = new HashMap<>();
        mTypeCount = 0;
    }

    public static synchronized DataUtil getInstance() {
        if (instance == null) {
            instance = new DataUtil();
        }
        return instance;
    }

    public Map<String, Integer> getmTypes() {
        return mTypes;
    }

    public int getmTypeCount() {
        return mTypeCount;
    }

    public String getTypeName(int type) {
        if (type == -1) {
            return new String("无法判断");
        }
        Iterator<String> keys = mTypes.keySet().iterator();//keyset()返回key
        while (keys.hasNext()) {
            String key = keys.next();
            if (mTypes.get(key) == type) {
                return key;
            }
        }
        return null;
    }

    //根据文件生成训练集
    public List<DataNode> getDataList(String fileName, String sep) throws Exception {
        List<DataNode> list = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(new File(fileName)));
        String line = null;
        while ((line = br.readLine()) != null) {
            String splits[] = line.split(sep);
            DataNode node = new DataNode();
            int i = 0;
            for (; i < splits.length; i++) {
                try {
                    node.addmAttribList(Float.valueOf(splits[i]));
                } catch (NumberFormatException e) {//非数字 为类别 否则映射为数字
                    if (!mTypes.containsKey(splits[i])) {
                        mTypes.put(splits[i], mTypeCount);
                        mTypeCount++;
                    }
                    node.setType(mTypes.get(splits[i]));
                    list.add(node);
                }
            }
        }
        return list;
    }
}
