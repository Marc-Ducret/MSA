package edu.usc.thevillagers.serversideagent.recording.event;

import com.mojang.authlib.GameProfile;

import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayer;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordWorker;
import net.minecraft.entity.Entity;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.util.math.AxisAlignedBB;
import net.minecraftforge.registries.GameData;

/**
 * Record of a new entity.
 */
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
			e = wr.createReplayEntityPlayer(wr.world, 
					new GameProfile(data.getUniqueId("ProfileUUID"), data.getString("ProfileName")));
		else
			e = GameData.getEntityRegistry().getValue(type).newInstance(wr.world);
		wr.spawnEntity(id, e);
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

	@Override
	public boolean isWithinBounds(WorldRecordWorker record, AxisAlignedBB bounds) {
		return true;
	}
}
