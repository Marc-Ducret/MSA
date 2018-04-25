package edu.usc.thevillagers.serversideagent.recording.event;

import java.util.UUID;

import com.mojang.authlib.GameProfile;

import edu.usc.thevillagers.serversideagent.ServerSideAgentMod;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayer;
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
	public void replay(WorldRecordReplayer wr) {
		Entity e;
		if(type < 0) 
			e = ServerSideAgentMod.proxy.createReplayEntityPlayer(wr.world.fakeWorld, 
					new GameProfile(UUID.randomUUID(), "Player")); //TODO get real profile?
		else
			e = EntityList.createEntityByID(type, wr.world.fakeWorld);
		e.setEntityId(id);
		wr.world.spawnEntity(e);
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
