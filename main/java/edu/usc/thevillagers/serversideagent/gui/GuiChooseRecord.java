package edu.usc.thevillagers.serversideagent.gui;

import java.io.File;
import java.io.IOException;

import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.Gui;
import net.minecraft.client.gui.GuiListExtended;
import net.minecraft.client.gui.GuiListExtended.IGuiListEntry;
import net.minecraft.client.gui.GuiScreen;

/**
 * Displays a list of records found on disk.
 */
public class GuiChooseRecord extends GuiScreen {
	
	private GuiRecordList list;
	
	@Override
	public void initGui() {
		super.initGui();
		
		list = new GuiRecordList(new File("tmp/records"), mc, width, height, 32, height - 64, 12);
	}

	@Override
	public void drawScreen(int mouseX, int mouseY, float partialTicks) {
		drawDefaultBackground();
		list.drawScreen(mouseX, mouseY, partialTicks);
		super.drawScreen(mouseX, mouseY, partialTicks);
	}
	
	@Override
	public void handleMouseInput() throws IOException {
		super.handleMouseInput();
		list.handleMouseInput();
	}
	
	@Override
	protected void mouseClicked(int mouseX, int mouseY, int mouseButton) throws IOException {
		super.mouseClicked(mouseX, mouseY, mouseButton);
		list.mouseClicked(mouseX, mouseY, mouseButton);
	}
	
	@Override
	protected void mouseReleased(int mouseX, int mouseY, int state) {
		super.mouseReleased(mouseX, mouseY, state);
		list.mouseReleased(mouseX, mouseY, state);
	}
	
	private static class GuiRecordList extends GuiListExtended {
		
		private GuiRecordEntry[] entries;
		
		public GuiRecordList(File folder, Minecraft mcIn, int widthIn, int heightIn, int topIn, int bottomIn, int slotHeightIn) {
			super(mcIn, widthIn, heightIn, topIn, bottomIn, slotHeightIn);
			File[] files = folder.listFiles(
					(file) -> 
						file.isDirectory() && 
						file.listFiles((f) -> f.getName().equals("record.info")).length > 0);
			entries = new GuiRecordEntry[files.length];
			for(int i = 0; i < entries.length; i ++)
				entries[i] = new GuiRecordEntry(files[i]);
		}
		
		@Override
		protected int getSize() {
			return entries.length;
		}

		@Override
		public IGuiListEntry getListEntry(int index) {
			return index >= 0 && index < entries.length ? entries[index] : null;
		}
	}
	
	private static class GuiRecordEntry extends Gui implements IGuiListEntry {

		private final File recordFolder;
		
		public GuiRecordEntry(File recordFolder) {
			this.recordFolder = recordFolder;
		}
		
		@Override
		public void updatePosition(int slotIndex, int x, int y, float partialTicks) {
		}

		@Override
		public void drawEntry(int slotIndex, int x, int y, int listWidth, int slotHeight, int mouseX, int mouseY,
				boolean isSelected, float partialTicks) {
			drawString(Minecraft.getMinecraft().fontRenderer, recordFolder.getName(), x, y, isSelected ? 0xFFFFFF : 0x808080);
		}

		@Override
		public boolean mousePressed(int slotIndex, int mouseX, int mouseY, int mouseEvent, int relativeX,
				int relativeY) {
			Minecraft.getMinecraft().addScheduledTask(() -> Minecraft.getMinecraft().displayGuiScreen(new GuiReplay(recordFolder)));
			return true;
		}

		@Override
		public void mouseReleased(int slotIndex, int x, int y, int mouseEvent, int relativeX, int relativeY) {
		}
	}
}