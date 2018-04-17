package edu.usc.thevillagers.serversideagent.recording;

import net.minecraft.nbt.NBTTagCompound;

public abstract class RecordEvent {

	public abstract void replay(WorldRecord wr);
	public abstract void write(NBTTagCompound compound);
	public abstract void read(NBTTagCompound compound);
	
	private static final Class<?>[] classes = new Class<?>[] {
		RecordEventBlockChange.class
	};
	
	public static Class<?> getClassFromId(int id) {
		if(id < 0 || id >= classes.length) throw new IllegalArgumentException("Unknown event id "+id);
		return classes[id];
	}
	
	public static int getClassId(Class<?> c) {
		for(int id = 0; id < classes.length; id++)
			if(classes[id] == c)
				return id;
		throw new IllegalArgumentException("Unknown event class "+c);
	}
	
	public static RecordEvent instantiate(int id) {
		try {
			return (RecordEvent) getClassFromId(id).newInstance();
		} catch (InstantiationException | IllegalAccessException e) {
			throw new IllegalArgumentException("Can't instantiate event with id "+id, e);
		}
	}
}
