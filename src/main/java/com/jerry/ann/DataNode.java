package com.jerry.ann;

import java.util.ArrayList;
import java.util.List;

public class DataNode {
    private List<Float> mAttribList;

    public List<Float> getmAttribList() {
        return mAttribList;
    }

   public void addmAttribList(Float val)
   {
       mAttribList.add(val);
   }

    public int getType() {
        return type;
    }

    public void setType(int type) {
        this.type = type;
    }

    private int type;
    public DataNode()
    {
        mAttribList=new ArrayList<>();
    }
}
