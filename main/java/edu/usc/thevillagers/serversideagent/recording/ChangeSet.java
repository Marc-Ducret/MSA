package edu.usc.thevillagers.serversideagent.recording;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import edu.usc.thevillagers.serversideagent.recording.event.RecordEvent;
import net.minecraft.nbt.NBTBase;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.nbt.NBTTagList;
import net.minecraftforge.common.util.Constants;

/**
 * File interface for a set of changes across multiple ticks.
 */
public class ChangeSet extends NBTFileInterface<List<List<RecordEvent>>> {
	
	public ChangeSet(File file) {
		super(file);
	}
	
	public void resetEventList() {
		data = new ArrayList<>();
	}
	
	public void appendChanges(List<RecordEvent> events) {
		data.add(events);
	}

	@Override
	protected void writeNBT(NBTTagCompound compound) throws IOException {
		NBTTagList listList = new NBTTagList();
		for(List<RecordEvent> evs : data) {
			NBTTagList listEvent = new NBTTagList();
			for(RecordEvent e : evs) {
				listEvent.appendTag(RecordEvent.toNBT(e));
			}
			listList.appendTag(listEvent);
		}
		compound.setTag("List", listList);
	}

	@Override
	protected void readNBT(NBTTagCompound compound) throws IOException {
		data = new ArrayList<>();
		NBTTagList listList = compound.getTagList("List", Constants.NBT.TAG_LIST);
		for(NBTBase nbt : listList) {
			NBTTagList listEvent = (NBTTagList) nbt;
			List<RecordEvent> evs = new ArrayList<>();
			for(NBTBase nbtEv : listEvent) {
				evs.add(RecordEvent.fromNBT((NBTTagCompound) nbtEv));
			}
			data.add(evs);
		}
	}
}
