package edu.usc.thevillagers.serversideagent.proxy;

import org.lwjgl.input.Keyboard;

import edu.usc.thevillagers.serversideagent.gui.GuiChooseRecord;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.GuiScreen;
import net.minecraftforge.client.event.GuiScreenEvent;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;

public class ClientProxy implements Proxy {
	
	@SubscribeEvent
    public void eventGui(GuiScreenEvent.KeyboardInputEvent event) {
    	char c = Keyboard.getEventCharacter();
    	if(c == 18 && GuiScreen.isCtrlKeyDown()) {
    		event.setCanceled(true);
    		Minecraft.getMinecraft().addScheduledTask(() -> Minecraft.getMinecraft().displayGuiScreen(new GuiChooseRecord()));
    	}
    }
}
