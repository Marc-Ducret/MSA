package edu.usc.thevillagers.serversideagent.gui;

import java.util.UUID;

import org.lwjgl.input.Keyboard;
import org.lwjgl.input.Mouse;
import org.lwjgl.opengl.GL11;
import org.lwjgl.util.glu.Project;

import com.mojang.authlib.GameProfile;

import net.minecraft.block.state.IBlockState;
import net.minecraft.client.gui.GuiScreen;
import net.minecraft.client.multiplayer.WorldClient;
import net.minecraft.client.network.NetHandlerPlayClient;
import net.minecraft.client.renderer.BufferBuilder;
import net.minecraft.client.renderer.GlStateManager;
import net.minecraft.client.renderer.Tessellator;
import net.minecraft.client.renderer.texture.TextureMap;
import net.minecraft.client.renderer.vertex.DefaultVertexFormats;
import net.minecraft.init.Blocks;
import net.minecraft.network.EnumPacketDirection;
import net.minecraft.network.NetworkManager;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.EnumDifficulty;
import net.minecraft.world.GameType;
import net.minecraft.world.WorldSettings;
import net.minecraft.world.WorldType;

public class GuiReplay extends GuiScreen {
	
	private WorldClient world;
	
	private Vec3d camPos = Vec3d.ZERO.addVector(0, 10, 0), prevCamPos = camPos;
	private float camYaw = 0, prevCamYaw = camYaw;
	private float camPitch = -90, prevCamPitch = camPitch;

	@Override
	public void initGui() {
		super.initGui();
		System.out.println("New gui replay");
		try {
			WorldSettings settings = new WorldSettings(0, GameType.NOT_SET, false, false, WorldType.FLAT);
			GameProfile profile = new GameProfile(UUID.randomUUID(), "dummy");
			NetHandlerPlayClient nethandler = new NetHandlerPlayClient(mc, this, new NetworkManager(EnumPacketDirection.CLIENTBOUND), profile);
			world = new WorldClient(nethandler, settings, 0, EnumDifficulty.PEACEFUL, mc.mcProfiler);
			Mouse.setGrabbed(true);
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public void updateScreen() {
		super.updateScreen();
		Vec3d move = Vec3d.ZERO;
		if(Keyboard.isKeyDown(mc.gameSettings.keyBindForward.getKeyCode())) move = move.addVector(+0, +0, -1);
		if(Keyboard.isKeyDown(mc.gameSettings.keyBindBack	.getKeyCode())) move = move.addVector(+0, +0, +1);
		if(Keyboard.isKeyDown(mc.gameSettings.keyBindRight	.getKeyCode())) move = move.addVector(+1, +0, +0);
		if(Keyboard.isKeyDown(mc.gameSettings.keyBindLeft	.getKeyCode())) move = move.addVector(-1, +0, +0);
		if(Keyboard.isKeyDown(mc.gameSettings.keyBindJump	.getKeyCode())) move = move.addVector(+0, +1, +0);
		if(Keyboard.isKeyDown(mc.gameSettings.keyBindSneak	.getKeyCode())) move = move.addVector(+0, -1, +0);
		move = move.normalize();
		prevCamPos = camPos;
		prevCamYaw = camYaw;
		prevCamPitch = camPitch;
		
		camPos = camPos.add(move.rotateYaw((float)Math.toRadians(-camYaw)).scale(.5));
		
		camYaw += Mouse.getDX() * .2;
		camPitch += Mouse.getDY() * .2;
	}
	
	@Override
	public void drawScreen(int mouseX, int mouseY, float partialTicks) {
		super.drawScreen(mouseX, mouseY, partialTicks);
		drawDefaultBackground();
		
		try {
			IBlockState state = Blocks.GLASS.getDefaultState();
			BlockPos pos = BlockPos.ORIGIN;
			world.setBlockState(pos, state);
			state = state.getActualState(world, pos);
			
			GlStateManager.matrixMode(GL11.GL_PROJECTION);
            GlStateManager.loadIdentity();
            Project.gluPerspective(90, (float)this.mc.displayWidth / (float)this.mc.displayHeight, 0.05F, 1000 * 2.0F);
            
            GlStateManager.matrixMode(GL11.GL_MODELVIEW);
			
			GlStateManager.pushMatrix();
			GlStateManager.loadIdentity();
			GlStateManager.color(1.0F, 1.0F, 1.0F, 1.0F);
	        GlStateManager.translate(0, 0.0F, 0F);
			
			mc.renderEngine.bindTexture(TextureMap.LOCATION_BLOCKS_TEXTURE);
			BufferBuilder buffer = Tessellator.getInstance().getBuffer();
			buffer.begin(7, DefaultVertexFormats.BLOCK);
			buffer.setTranslation(0, 0, 0);
			buffer.noColor();
			
			float curCamPitch = prevCamPitch + (camPitch - prevCamPitch) * partialTicks;
			float curCamYaw   = prevCamYaw	 + (camYaw   - prevCamYaw  ) * partialTicks;
			GlStateManager.rotate(-curCamPitch, 1, 0, 0);
			GlStateManager.rotate(curCamYaw, 0, 1, 0);
			Vec3d curCamPos = prevCamPos.add(camPos.subtract(prevCamPos).scale(partialTicks));
			GlStateManager.translate(-curCamPos.x, -curCamPos.y, -curCamPos.z);
			
			boolean result = mc.getBlockRendererDispatcher().renderBlock(state, pos, world, Tessellator.getInstance().getBuffer());
			Tessellator.getInstance().draw();
			
			GlStateManager.popMatrix();
			this.drawCenteredString(this.fontRenderer, ""+result, this.width / 2, 40, 16777215);
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public boolean doesGuiPauseGame() {
		return false;
	}
}