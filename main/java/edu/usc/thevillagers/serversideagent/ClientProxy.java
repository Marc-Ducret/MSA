package edu.usc.thevillagers.serversideagent;

import org.lwjgl.input.Keyboard;

import edu.usc.thevillagers.serversideagent.gui.GuiReplay;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.GuiScreen;
import net.minecraftforge.client.event.GuiScreenEvent;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;

public class ClientProxy {
	
	@SubscribeEvent
    public void eventGui(GuiScreenEvent.KeyboardInputEvent event) {
    	char c = Keyboard.getEventCharacter();
    	if(c == 18 && GuiScreen.isCtrlKeyDown()) {
    		event.setCanceled(true);
    		Minecraft.getMinecraft().addScheduledTask(() -> Minecraft.getMinecraft().displayGuiScreen(new GuiReplay()));
    	}
    }
}
