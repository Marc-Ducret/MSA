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
				NBTTagCompound nbtEvent = new NBTTagCompound();
				e.write(nbtEvent);
				nbtEvent.setInteger("Id", RecordEvent.getClassId(e.getClass()));
				listEvent.appendTag(nbtEvent);
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
				NBTTagCompound compEv = (NBTTagCompound) nbtEv;
				RecordEvent ev = RecordEvent.instantiate(compEv.getInteger("Id"));
				ev.read(compEv);
				evs.add(ev);
			}
			data.add(evs);
		}
	}
}
