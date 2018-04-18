package edu.usc.thevillagers.serversideagent.recording.event;

import java.util.UUID;

import com.mojang.authlib.GameProfile;

import edu.usc.thevillagers.serversideagent.recording.WorldRecord;
import net.minecraft.client.entity.EntityOtherPlayerMP;
import net.minecraft.entity.Entity;
import net.minecraft.entity.EntityList;
import net.minecraft.nbt.NBTTagCompound;

public class RecordEventEntitySpawn extends RecordEvent {
	
	private int id, type;
	private NBTTagCompound data;
	
	public RecordEventEntitySpawn() {
	}
	
	public RecordEventEntitySpawn(int id, int type, NBTTagCompound data) {
		this.id = id;
		this.type = type;
		this.data = data;
	}

	@Override
	public void replay(WorldRecord wr) {
		Entity e;
		if(type < 0) e = new EntityOtherPlayerMP(wr.getReplayWorld().fakeWorld, new GameProfile(UUID.randomUUID(), "PlayerName"));
		else e = EntityList.createEntityByID(type, wr.getReplayWorld().fakeWorld);
		wr.getReplayWorld().spawnEntity(e);
		System.out.println(id+" SPAWN "+e);
		wr.entitiesData.put(id, data);
	}

	@Override
	public void write(NBTTagCompound compound) {
		compound.setInteger("Id", id);
		compound.setInteger("Type", type);
		compound.setTag("Data", data);
	}

	@Override
	public void read(NBTTagCompound compound) {
		id = compound.getInteger("Id");
		type = compound.getInteger("Type");
		data = compound.getCompoundTag("Data");
	}
}
