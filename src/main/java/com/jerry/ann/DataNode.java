package com.jerry.ann;

import java.util.ArrayList;
import java.util.List;

public class DataNode {
    //dataNode为每一行的数据 被封装在list之中 判断如果不是float则把他转化为float（为花卉的类型）
    private List<Float> mAttribList;

    public List<Float> getmAttribList() {
        return mAttribList;
    }

    public void addmAttribList(Float val) {
        mAttribList.add(val);
    }

    public int getType() {
        return type;
    }

    public void setType(int type) {
        this.type = type;
    }

    private int type;

    public DataNode() {
        mAttribList = new ArrayList<>();
    }
}
